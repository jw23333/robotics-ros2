from geometry_msgs.msg import PoseArray, Pose, Quaternion
import random
from sklearn.cluster import KMeans
from .pf_base import PFLocaliserBase
from .util import rotateQuaternion, getHeading
from copy import deepcopy
import numpy as np

class PFLocaliser(PFLocaliserBase):

    def __init__(self, logger, clock):
        # Call the superclass constructor
        super().__init__(logger, clock)

        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.5
        self.ODOM_TRANSLATION_NOISE = 0.5
        self.ODOM_DRIFT_NOISE = 0.5
        self.weights = []

        

    def initialise_particle_cloud(self, initialpose=None):
        """
        Initialize the particle cloud around the given initial pose with added noise,
        or at a random location on the map if initialpose is None.
        """
        num_particles = 100
        self.particlecloud = PoseArray()
        self.particlecloud.header.frame_id = "map"

        for i in range(num_particles):
            particle = Pose()
            if initialpose:
                # Initialize around the given initial pose with noise
                particle.position.x = random.gauss(initialpose.pose.pose.position.x, self.ODOM_TRANSLATION_NOISE)
                particle.position.y = random.gauss(initialpose.pose.pose.position.y, self.ODOM_TRANSLATION_NOISE)
            else:
                # Initialize at a random location on the map
                particle.position.x = random.uniform(0, self.occupancy_map.info.width * self.occupancy_map.info.resolution)
                particle.position.y = random.uniform(0, self.occupancy_map.info.height * self.occupancy_map.info.resolution)
            
            particle.position.z = 0.0  # Assume a 2D environment
            random_yaw = random.uniform(-np.pi, np.pi)
            particle.orientation = rotateQuaternion(Quaternion(w=1.0), random_yaw)

            self.particlecloud.poses.append(particle)
        return self.particlecloud

    def update_particle_cloud(self, scan):
        """
        Update the particle cloud using the sensor model.
        The sensor model updates the particle weights based on the laser scan data.
        """
        # Step 1: Apply the sensor model to update particle weights based on the laser scan
        # Step 1: Apply the sensor model to update particle weights based on the laser scan
        self.weights = []
        for particle in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, particle)
            self.weights.append(weight)

        # Check if weights array is empty
        if not self.weights:
            self.logger.error("Weights array is empty. Skipping update.")
            return

        average_weight = sum(self.weights) / len(self.weights)
        standard_deviation = np.std(self.weights)
        print("Average weight: ", average_weight)
        print("Standard deviation: ", standard_deviation)
        if standard_deviation < 1.5 or average_weight < 3.15:
            self.add_new_particle()
        
        # Normalize the weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            # If all weights are zero, assign equal weight to all particles
            self.weights = [1.0 / len(self.weights)] * len(self.weights)
        
        self.resample_particles(self.weights)
    
    
    def resample_particles(self, weights):
        """
        Resample particles based on their weights using systematic resampling.
        """
        new_particles = []
        num_particles = len(self.particlecloud.poses)
        positions = [(i + random.uniform(0, 1)) / num_particles for i in range(num_particles)]
        cumulative_sum = [sum(weights[:i+1]) for i in range(num_particles)]
        i, j = 0, 0
        while i < num_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles.append(deepcopy(self.particlecloud.poses[j]))
                i += 1
            else:
                j += 1
        for particle in new_particles:
            #add gaussian noise to the resampled particles for jittering
            particle.position.x += random.gauss(0, 0.1)
            particle.position.y += random.gauss(0, 0.1)
            particle.position.z = 0.0
            particle.orientation = rotateQuaternion(particle.orientation, random.gauss(0, 0.1))
        self.particlecloud.poses = new_particles

    def add_new_particle(self):
        """
        Replace the particles with the lowest weights with new randomly generated particles.
        """
        num_particles_to_replace = 93  # Number of particles to replace
        def lower_bound(x):
            return -0.85*x-1
        def upper_bound(x):
            return -0.85*x+4
        # Step 1: Sort particles by weight in ascending order
        sorted_indices = np.argsort(self.weights)
    
        # Step 2: Replace the particles with the lowest weights
        for i in range(num_particles_to_replace):
            index_to_replace = sorted_indices[i]
            particle = Pose()
            particle.position.x = random.uniform(-8, 12)
            particle.position.y = random.uniform(lower_bound(particle.position.x), upper_bound(particle.position.x))
            particle.position.z = 0.0
            random_yaw = random.uniform(-np.pi, np.pi)
            particle.orientation = rotateQuaternion(Quaternion(w=1.0), random_yaw)
            self.particlecloud.poses[index_to_replace] = particle

    def estimate_pose(self):
        """
        Estimate the robot's pose based on the particle cloud by averaging the positions and orientations of all particles.
        """
        num_particles = len(self.particlecloud.poses)
    
        # Extract positions and orientations
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in self.particlecloud.poses])
        orientations = np.array([[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in self.particlecloud.poses])
    
        # Calculate the mean position and orientation of all particles
        mean_position = np.mean(positions, axis=0)
        mean_orientation = np.mean(orientations, axis=0)
        norm = np.linalg.norm(mean_orientation)
        mean_orientation /= norm
    
        # Create and return the estimated pose
        estimated_pose = Pose()
        estimated_pose.position.x = mean_position[0]
        estimated_pose.position.y = mean_position[1]
        estimated_pose.position.z = mean_position[2]
        estimated_pose.orientation.x = mean_orientation[0]
        estimated_pose.orientation.y = mean_orientation[1]
        estimated_pose.orientation.z = mean_orientation[2]
        estimated_pose.orientation.w = mean_orientation[3]
    
        return estimated_pose