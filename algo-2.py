import numpy as np
from sympy import symbols, Eq, solve
import tensorflow as tf
from tqdm import tqdm
import os
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import sys
import time



def load_and_update_model(mass):
    global model  

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    MODEL_PATH = f"/home/karen.kang/LIGOSURF23/network_training/models/models/mismatch_allmodes_{mass}"

    # Load the model and update the global variable
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def convert_to_lambda(q, spin1, spin2, theta1, theta2):
    eta = q / (1 + q)**2
    phi1, phi2 = np.random.uniform(0, 2 * np.pi, 2)
    a1z = spin1 * np.cos(theta1)
    a2z = spin2 * np.cos(theta2)
    a1x = spin1 * np.sin(theta1) * np.cos(phi1)
    a1y = spin1 * np.sin(theta1) * np.sin(phi1)
    a2x = spin2 * np.sin(theta2) * np.cos(phi2)
    a2y = spin2 * np.sin(theta2) * np.sin(phi2)

    return [eta, a1x, a1y, a1z, a2x, a2y, a2z]



class BBH:
    def __init__(self, lam = None):
        if lam is None:
            self.random()
        else:
            self.lam = lam
            self.initialize_from_lambda()
            
        self.z1 = self.a1[2]
        self.z2 = self.a2[2]
            
    def random(self):
        #eta, q
        self.initialize_mass_ratio()
        #a1,a2, spin1, spin2 (theta1, theta2, inplane1, inplane2 if precession)
        self.construct_spin_vectors()
        assert 0 <= self.spin1 <= 1, "spin vector 1 too large: spin1 must be between 0 and 1."
        assert 0 <= self.spin2 <= 1, "spin vector 2 too large: spin2 must be between 0 and 1."
        self.find_chi_eff()
        
        self.construct_features_array()
        self.find_chi_p()
            


    def initialize_from_lambda(self):
        assert len(self.lam) == 7, "The 'lam' list must contain exactly 7 entries."

        self.eta = self.lam[0]
        self.q_from_eta()
        assert 1 <= self.q <= 10, "mass ratio out of bounds"

        self.a1 = self.lam[1:4]
        self.a2 = self.lam[4:7]
        self.z1 = self.a1[2]
        self.z2 = self.a2[2]
        
        self.spin1 = find_spin_mag(self.a1)
        self.spin2 = find_spin_mag(self.a2)

        assert 0 <= self.spin1 <= 1, "spin vector 1 too large: spin1 must be between 0 and 1."
        assert 0 <= self.spin2 <= 1, "spin vector 2 too large: spin2 must be between 0 and 1."


        self.inplane1 = find_spin_mag(self.a1[:2])
        self.inplane2 = find_spin_mag(self.a2[:2])
        self.theta1 = find_theta(self.z1,self.spin1)
        self.theta2 = find_theta(self.z2,self.spin2)
        self.find_chi_p()
        
        self.theta1 = 0 if np.isnan(self.theta1) else self.theta1
        self.theta2 = 0 if np.isnan(self.theta2) else self.theta2
        
        self.find_chi_eff()

    def initialize_mass_ratio(self):
        self.q = np.random.uniform(1,10)
        self.eta = self.q/(self.q+1)**2

    def construct_spin_vectors(self):
        self.a1, self.spin1, self.theta1 = self.random_spin_vector()
        self.a2, self.spin2, self.theta2 = self.random_spin_vector()
        self.inplane1 = find_spin_mag(self.a1[:2])
        self.inplane2 = find_spin_mag(self.a2[:2])
     

    def construct_features_array(self):
        self.lam =[self.eta]+self.a1+self.a2


    def find_chi_p(self):
        frac = (4+3*self.q)/(4*self.q**2+3*self.q)
        self.chi_p = max(self.spin1, frac*self.spin2)
        assert 0 <= self.chi_p <= 1, "chi_p out of bounds"


    def find_chi_eff(self):
        self.chi_eff = (self.q*self.a1[2]+self.a2[2])/(self.q+1)

        assert -1 <= self.chi_eff <= 1, "chi_eff out of bounds"


    def q_from_eta(self):
        q = symbols('q')
        equation = Eq(self.eta, q / (q + 1)**2)
        solutions = solve(equation, q)
        valid_solutions = [sol for sol in solutions if sol >= 1]
        self.q = float(valid_solutions[0])

    def random_spin_vector(self):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, 1)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return [x, y, z], r, theta


def find_spin_mag(vec):
    mag = np.linalg.norm(vec)
    return mag

def find_theta(z, spin):
    assert 0<= np.abs(z)<=spin, "Aligned spin mag should always be between 0 and spin magnitude"
    theta = np.arccos(z/spin)
    # if theta < 0:
    #     theta = np.pi + theta
    return theta

def find_inplane(theta,spin):
    inplane = spin * np.sin(theta)
    return inplane


class ParameterSpace:
    PARAM_ATTRIBUTES = {
        "eta": 'eta',
        "q": 'q',
        "chieff": 'chi_eff',
        "spin1": 'spin1',
        "spin2": 'spin2',
        "theta1": 'theta1',
        "theta2": 'theta2',
        "mismatch": 'mismatch',
        "chip": 'chi_p',
        "z1": 'z1',
        "z2": 'z2'
    }

    PARAM_STRINGS = {
        "eta": "$\eta$",
        "q": "$q$",
        "chieff": "$\chi_{\mathrm{eff}}$",
        "spin1": "$a_{1}$",
        "spin2": "$a_{2}$",
        "theta1": "$\\theta_1$",
        "theta2": "$\\theta_2$",
        "mismatch": "$\mathcal{MM}$",
        "chip": "$\chi_{p}$",
        "z1": '$a_{1z}$',
        "z2": '$a_{2z}$'
    }

    PARAM_BOUNDS = {
        "eta": (0.0826, 0.25),
        "q": (1, 10),
        "chieff": (-1, 1),
        "spin1": (0, 1),
        "spin2": (0, 1),
        "theta1": (0, 1),
        "theta2": (0, 1),
        "mismatch": (0, 1),
        "chip": (0, 1),
        "z1": (-1,1),
        "z2": (-1,1)
    }


    def __init__(self, sample=10000, lam0 = None, inj = False, data = None, start = None):
        
        self.lam0 = lam0
        self.sample = sample
        self.inj = inj
        self.bbh_instances = []  # List to store BBH instances
                
        if self.lam0 is not None:
            bbh0 = BBH(lam = lam0)
            self.bbh_instances.append(bbh0)
            features = self.lam0 + self.lam0
            sample = sample - 1
            
        if self.inj: 
            for _ in tqdm(range(sample)):
                self.data = data
                self.start = start
                perturbed_lam = self.perturb_lam()
                bbh = BBH(lam=perturbed_lam)  
                self.bbh_instances.append(bbh)
                          
        else:
            for _ in tqdm(range(0, sample)):
                bbh = BBH()
                self.bbh_instances.append(bbh)
                
     
        # Extract data for easy access
        self.eta = np.array([bbh.eta for bbh in self.bbh_instances])
        self.q = np.array([bbh.q for bbh in self.bbh_instances])
        self.chi_eff = np.array([bbh.chi_eff for bbh in self.bbh_instances])
        self.spin1 = np.array([bbh.spin1 for bbh in self.bbh_instances])
        self.spin2 = np.array([bbh.spin2 for bbh in self.bbh_instances])
        self.z1 = np.array([bbh.z1 for bbh in self.bbh_instances])
        self.z2 = np.array([bbh.z2 for bbh in self.bbh_instances])
        self.chi_p = np.array([bbh.chi_p for bbh in self.bbh_instances])
        self.theta1 = np.array([bbh.theta1 for bbh in self.bbh_instances])
        self.theta2 = np.array([bbh.theta2 for bbh in self.bbh_instances])
 
        self.lams = np.array([bbh.lam for bbh in self.bbh_instances])
        
        
        features_list = []
        
        if self.lam0 is not None:
            for lam in self.lams:
                f = np.concatenate((lam0, lam))
                features_list.append(f)
            
            features = np.array(features_list)
            self.mismatch = model.predict(features).flatten()
                   

        self.summary()

    def summary(self):
        if self.lam0 is not None:
            values_and_features = {
                'eta': self.eta,
                'q': self.q,
                'chieff': self.chi_eff,
                'chip': self.chi_p,
                'spin1': self.spin1,
                'spin2': self.spin2,
                'theta1': self.theta1,
                'theta2': self.theta2,
                'z1': self.z1,
                'z2': self.z2,
                'mismatch': self.mismatch,
            }
        else:
            values_and_features = {
                'eta': self.eta,
                'q': self.q,
                'chieff': self.chi_eff,
                'chip': self.chi_p,
                'spin1': self.spin1,
                'spin2': self.spin2,
                'theta1': self.theta1,
                'theta2': self.theta2 ,
                'z1': self.z1,
                'z2': self.z2
            }    
        return values_and_features
    
        

    def perturb_lam(self):
        sampled_lams = []
       
        # Determine bounds from the data
        min_bounds = np.min(self.data, axis=0)
        max_bounds = np.max(self.data, axis=0)

        # Perturb the starting parameters within the bounds
        perturbed_params = [np.random.uniform(low=min_bound, high=max_bound) for min_bound, max_bound, start_val in zip(min_bounds, max_bounds, self.start)]

        # Convert the perturbed parameters to lambda representation
        lambda_sample = convert_to_lambda(*perturbed_params)

        return lambda_sample
    
    
    
#     def pick_bbh_kN(self, target_params, direction_vector, num_points=35, step_size=0.1):
#         """
#         Finds the nearest bbh instance to the target parameters using K-NN.

#         Args:
#         - target_params (dict): Dictionary of target parameters.

#         Returns:
#         - The lam value of the nearest bbh instance.
#         - List of actual values for the target parameters of the best match.
#         """
#         # Prepare the data for K-NN
#         data = []
#         for bbh_instance in self.bbh_instances:
#             instance_data = [getattr(bbh_instance, self.PARAM_ATTRIBUTES[param]) for param in target_params]
#             data.append(instance_data)

#         base_array = np.array([target_params[param] for param in target_params])
        
#         potential_targets = []
#         for i in range(num_points):
#             scaled_direction = i * step_size * direction_vector
#             new_target = base_array + scaled_direction
#             # print(type(scaled_direction), scaled_direction.shape)
#             potential_targets.append(new_target)

#         # Apply K-NN for the nearest neighbor
#         nbrs = NN(n_neighbors=1, algorithm='kd_tree').fit(data)
#         distances, indices = nbrs.kneighbors(potential_targets)
        
#         closest_target_idx = np.argmin(distances)

#         # Retrieve the best matching bbh instance
#         best_match_idx = indices[closest_target_idx][0]
#         best_match = self.bbh_instances[best_match_idx]


#         # Print the results
#         best_match_values = []
#         for param in target_params:
#             value = getattr(best_match, self.PARAM_ATTRIBUTES[param])
#             best_match_values.append(value)
#         print('point nearest to proposed:', best_match_values)
#         print("lam:", best_match.lam)

#         # Return lam and best match param values
#         return best_match.lam, best_match_values



class MapDegeneracyND:
    PARAM_BOUNDS = {
        "eta": (0.0826, 0.25),
        "q": (1, 10),
        "chieff": (-1, 1),
        "spin1": (0, 1),
        "spin2": (0, 1),
        "theta1": (0, np.pi),
        "theta2": (0, np.pi),
        "mismatch": (0, 1),
        "chip": (0, 1),
        "z1": (-1,1),
        "z2": (-1,1)
        
    }
    

    PARAM_STRINGS = {
        "eta": "$\eta$",
        "q": "$q$",
        "chieff": "$\chi_{\mathrm{eff}}$",
        "spin1": "$a_{1}$",
        "spin2": "$a_{2}$",
        "theta1": "$\\theta_1$",
        "theta2": "$\\theta_2$",
        "mismatch": "$\mathcal{MM}$",
        "chip": "$\chi_{p}$",
        "z1": '$a_{1z}$',
        "z2": '$a_{2z}$'
    }

    def __init__(
        self,
        lam0,
        start,
        dimensions=["eta", "chieff"],
        stepsize = 1,
        max_iterations=100,
        sample = 100000,
        fit_type="GMM",
        percentage = 1.,
        SNR = 4
    ):
        #initial BBH parameters
        self.start = start
        self.origin = start.copy()
        self.lam0 = lam0
        self.lam = lam0.copy()

        #mapping parameters
        self.dims = len(dimensions)
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.fit_type = fit_type
        self.step_size = stepsize
        self.sample = sample

        self.SNR = SNR
        self.percentage = percentage


        # Check if the provided dimensions list matches the dim argument
        if dimensions and len(dimensions) != self.dims:
            raise ValueError(f"The dimensions list provided does not match the specified dim of {dims}.")

        #values stored
        self.eigenvalues = []
        self.eigenvectors = []
        self.mismatch_from_reference = []
        self.mismatch_from_previous = []
        self.points = [np.array(self.start)]
        self.lams = [np.array(self.lam)]
        self.steps = []
        self.means = []
        

        self.ps, self.data_arrays = self.prepare_data()
        self.data_arrays = {k: v for k, v in self.data_arrays.items() if v is not None and len(v) > 0}



    def prepare_data(self):
        parameterspace = ParameterSpace(sample = self.sample)
        data = parameterspace.summary()
        data_arrays = {dim: self._fetch_array(data, dim) for dim in self.dimensions}
        return parameterspace, data_arrays

    def find_mismatch(self):
        features_list = []
        for lam in self.ps.lams:
            f = np.concatenate((self.lam, lam))
            features_list.append(f)
        features = np.array(features_list)
        mismatch = model.predict(features).flatten()
        return mismatch
    
    
    def _fetch_array(self, data, key):
        """Fetch data array using the provided key."""
        return data.get(key, [])

    def rejection_sampling(self, mismatch):    
        data_arrays = self.data_arrays
        # Calculate weights for rejection sampling based on mismatch values
        weights = np.exp(-self.SNR*self.SNR*mismatch*mismatch/2)
        data_arrays['mismatch'] = mismatch
        masked_data = {k: v for k, v in data_arrays.items()}
        samples = np.column_stack(tuple(masked_data.values()))
        
        #sampled_data_with_mismatch = rejection_sample(samples, weights)
        keep = weights > np.random.uniform(0, max(weights), weights.shape)
        sampled_data_with_mismatch = samples[keep]

        print("Size after rejection sampling:", len(sampled_data_with_mismatch))

        # Optionally, remove the mismatch column from the final data if it's no longer needed
        final_data = sampled_data_with_mismatch[:, :-1]-self.start
        
        return final_data, sampled_data_with_mismatch
    
    def rejection_cut(self, mismatch):    
        '''
        REJECTION THEN MASKING
        '''
        data_arrays = self.data_arrays
        # Calculate weights for rejection sampling based on mismatch values
        weights = 1 - mismatch
        data_arrays['mismatch'] = mismatch
        masked_data = {k: v for k, v in data_arrays.items()}
        samples = np.column_stack(tuple(masked_data.values()))
        
        sampled_data_with_mismatch = rejection_sample(samples, weights)
        print("Size after rejection sampling:", len(sampled_data_with_mismatch))

        # Extract mismatch values from the last column of the sampled data
        sampled_mismatch = sampled_data_with_mismatch[:, -1]

        # Calculate the cutoff index for the desired percentage
        num_points = len(mismatch)
        cutoff_index = int(num_points * (self.percentage / 100.0))

        # Sort the mismatch values to find the cutoff value
        sorted_indices = np.argsort(sampled_mismatch)
        cutoff_value = sampled_mismatch[sorted_indices[cutoff_index]]

        # Apply the mismatch cut to keep only the desired percentage of data
        final_data_with_mismatch = sampled_data_with_mismatch[sampled_mismatch <= cutoff_value]

        # Print details about the cut
        print("Cutoff value:", cutoff_value)
        print("Final data size after cut:", len(final_data_with_mismatch))

        # Optionally, remove the mismatch column from the final data if it's no longer needed
        final_data = final_data_with_mismatch[:, :-1]

        return final_data
    
    
    
    def rejection_only(self, mismatch):    
        '''
        REJECTION ONLY
        '''
        data_arrays = self.data_arrays
        # Calculate weights for rejection sampling based on mismatch values
        weights = 1 - mismatch
        data_arrays['mismatch'] = mismatch
        masked_data = {k: v for k, v in data_arrays.items()}
        samples = np.column_stack(tuple(masked_data.values()))
        
        sampled_data_with_mismatch = rejection_sample(samples, weights)
        print("Size after rejection sampling:", len(sampled_data_with_mismatch))

        final_data = sampled_data_with_mismatch[:, :-1]

        return final_data



    
#     def mask_data_by_percentage(
#         self, # Dictionary: {'x': x_data, 'y': y_data, ...}
#         mismatch # Percentage of points to keep
#     ):
        
#         '''
#         MASKING THEN REJECCTION
#         '''
#         data_arrays = self.data_arrays
        
        
#         # Calculate the index corresponding to the desired percentage
#         percentage = self.percentage
#         num_points = len(mismatch)
#         cutoff_index = int(num_points * (percentage / 100.0))


#         # Sort the mismatch array and get the cutoff value for the desired percentage
#         sorted_indices = np.argsort(mismatch)
#         cutoff_value = mismatch[sorted_indices[cutoff_index]]

#         # Create a mask for values less than or equal to the cutoff value
#         mask = mismatch <= cutoff_value
        
#         data_arrays['mismatch'] = mismatch
#         # Apply the mask to each data array in data_arrays
#         masked_data = {k: v[mask] for k, v in data_arrays.items()}

#         # Stack the masked data arrays
#         data_to_fit = np.column_stack(tuple(masked_data.values()))[:, :-1]
        
#         weights = 1 - masked_data['mismatch']
#         posterior = rejection_sample(data_to_fit, weights)
        
#         # MEAN SUBTRACT!!!!!!!!
#         posterior = posterior -self.start

#         print("Cutoff value:", cutoff_value)
#         print("Weighted posterior size:", len(posterior))

#         return posterior


    def fit_data(self, data_to_fit):
        if self.fit_type == "GMM":
            g = BayesianGaussianMixture(
                n_components=1, covariance_type="tied", init_params="random_from_data"
            )
            GMM = g.fit(data_to_fit)
            # print("Means:", GMM.means_)
            self.means.append(GMM.means_)
            return GMM
        elif self.fit_type == "PCA":
            # Adjust the PCA components to match the number of dimensions
            pca = PCA(n_components=self.dims)
            PCA_model = pca.fit(data_to_fit)
            # print("Means:", PCA_model.mean_)
            self.means.append(PCA_model.mean_)
            return PCA_model

                
                
    def find_direction(self, fitted_data, **kwargs):
        if self.fit_type == "GMM":
            eig_val, eig_vec = np.linalg.eigh(fitted_data.covariances_)
            order = np.argsort(eig_val)[::-1]
            eig_val = eig_val[order]
            eig_vec = eig_vec[:, order]
            self.eigenvectors.append(eig_vec.T[0])
     
        elif self.fit_type == "PCA":
            eig_val = fitted_data.explained_variance_
            eig_vec = fitted_data.components_
            self.eigenvectors.append(eig_vec[0])

            
        # for i, (val, vec) in enumerate(zip(eig_val, eig_vec.T)):
        #     print(f"Eigenvalue {i + 1}: {val}")
        #     print(f"Corresponding eigenvector: {vec}\n")

        self.eigenvalues.append(eig_val)
        
        # Only keep largest
        eig_val = [eig_val[0]]
        if self.fit_type == "GMM":
            eig_vec = [eig_vec.T[0]]
        else:
            eig_vec = [eig_vec[0]]
         
        
        new_vector = np.sum([val * vec for val, vec in zip(eig_val, eig_vec)], axis=0)

        if self.steps:
            previous_step = self.steps[-1]  # Get the last step

            # Check the direction of each component
            for i in range(len(new_vector)):
                if previous_step[i] * new_vector[i] < 0:  # If the product is negative, they are in opposite directions
                    new_vector[i] = -new_vector[i]  # Invert the direction of the new step's component

        self.steps.append(new_vector)
        
        return new_vector



    def run_mapping(self, backward = False):
        multplier = 1
        
        if backward:
            multplier = -1
            
        first_iteration = True 
        
        ps = self.ps
        
        for iteration in range(self.max_iterations):
            try:
                mismatch = self.find_mismatch()

                # 2. Fit the data (GMM or PCA)  
                
                data_to_fit, full_data = self.rejection_sampling(mismatch)
                # data_to_fit = self.rejection_cut(mismatch)
                
                # if self.percentage is not None:
                #     if self.rejection and first_iteration:
                #         print('rejection first:')
                #         data_to_fit = self.rejection_cut(mismatch)
                #         # first_iteration = False
                #     else:    
                #         data_to_fit = self.rejection_only(mismatch)

                fitted_data = self.fit_data(data_to_fit)
                
                # either FORWARD or BACKWARD
                direction = self.find_direction(fitted_data)*multplier
                print("mapping to direction:", direction)
                
                
                next_point = self.start + (direction / np.sqrt(self.dims))*self.step_size
                print("currently at ", self.start)
                print("proposed start at ", next_point)
                if not self.is_within_bounds(next_point):
                    print("Reached boundary. Stopping.")
                    break

                # set starting point for next iteration
                self.start = next_point

                # Update parameters, self.lam
                self.mismatch_at_point(direction)
                
                # update ps
#                 if self.dims == 5:
#                     print('sampling around injection')
#                     self.ps = ParameterSpace(sample = self.sample, inj = True, data = data_to_fit, start = self.start)
                
                    
            except NoCorrelationException as e:
                print(e)
                print("No correlation found. Stopping mapping.")
                break
    
    def run_mapping_bothways(self):
        print('####################################')
        print('MAPPING FORWARD DIRECTION')
        self.run_mapping()
        print('####################################')        
        print('MAPPING BACKWARD DIRECTION')
        self.start = self.origin
        self.run_mapping(backward = True)
        print("Mapping complete.")
        
     

    def guess_lam(self, direction):
        if self.dims == 2:
            eta = self.start[0]
            a1z = self.start[1]
            a2z = self.start[1]
            lam = [eta,0,0,a1z,0,0,a2z]
            # Uniform Priors
        elif self.dims == 3:
            # sets x,y to 0
            q = self.start[0]
            eta = q/(1+q)**2
            a1z = self.start[1]
            a2z = self.start[2]
            lam = [eta,0,0,a1z,0,0,a2z]

        elif self.dims == 5:
            q = self.start[0]
            spin1, spin2 = self.start[1], self.start[2]
            theta1, theta2 = self.start[3], self.start[4]

            lam = convert_to_lambda(q,spin1,spin2,theta1,theta2)
                
        return lam
        

    def mismatch_at_point(self, direction):
        lam_old = self.lam
        self.lam = self.guess_lam(direction)
        
        print("new start at ", self.start)
        self.points.append(self.start)
        self.lams.append(self.lam)

        features1 = self.lam0 + self.lam
        features2 = lam_old + self.lam
        mismatch_prediction_ref = model.predict(features1).flatten()[0]
        mismatch_prediction_pre = model.predict(features2).flatten()[0]
        print(f"predicted mismatch from reference: {mismatch_prediction_ref}")
        print(f"predicted mismatch from previous: {mismatch_prediction_pre}")
        print("-----")

        self.mismatch_from_reference.append(mismatch_prediction_ref)
        self.mismatch_from_previous.append(mismatch_prediction_pre)

    def is_within_bounds(self, point):
        for dim_identifier, value in zip(self.dimensions, point):
            if not (
                self.PARAM_BOUNDS[dim_identifier][0]
                <= value
                <= self.PARAM_BOUNDS[dim_identifier][1]
            ):
                print(f"{dim_identifier} is out of bounds with value {value}")
                return False
        return True


class NoCorrelationException(Exception):
    pass

