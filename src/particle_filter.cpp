/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 50;

  // Adding noise to the GPS measurement. 
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> angle_theta(theta, std[2]);

  // Create new particles with randomised coordinates
  for(int i = 0; i < num_particles; i++) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_x(gen);
    sample_theta = angle_theta(gen);
    Particle new_particle = {i, sample_x, sample_y, sample_theta, 1.0};
    particles.push_back(new_particle);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  // Noise distributions
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  

  for(Particle &particle : particles) {
    // For straight path
    if(fabs(yaw_rate) < 0.001) {
      particle.x += velocity * delta_t * cos(particle.theta); 
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
    // For curved path
      particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }

    // Adding noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  // Compare predicted to observations, and find min distance. Assign predicted.id to obs.id
  for(int i = 0; i < observations.size(); i++) {
    double min_dist = std::numeric_limits<double>::max();
    int predicted_id = -1;
    for(int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(distance < min_dist) {
        min_dist = distance;
        predicted_id = predicted[j].id;
      }
    }
    observations[i].id = predicted_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // Clear old weights before calculating new;
  weights.clear();
  
  // For each particle update the weight.
  for(Particle &particle : particles) {
    vector<LandmarkObs> transformed_observations;

    // Transform the local coordinates of an observation to map coordinates.
    for(LandmarkObs obs : observations) {
      double x_map_obs = homogenous_transform_x(particle.x, obs.x, obs.y, particle.theta);
      double y_map_obs = homogenous_transform_y(particle.y, obs.x, obs.y, particle.theta);
      LandmarkObs transformed_obs = {obs.id, x_map_obs, y_map_obs};
      transformed_observations.push_back(transformed_obs);
    }
    
    // Store all the landmarks that are within 50m.
    vector<LandmarkObs> map_landmarks_filtered;
    for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      float map_landmark_x = map_landmarks.landmark_list[i].x_f;
      float map_landmark_y = map_landmarks.landmark_list[i].y_f;
      if(dist(particle.x, particle.y, map_landmark_x, map_landmark_y) <= sensor_range) {
        LandmarkObs landmark_in_range = {map_landmarks.landmark_list[i].id_i, map_landmark_x, map_landmark_y};
        map_landmarks_filtered.push_back(landmark_in_range);
      }
    }

    // Associate the observations to the landmarks.
    dataAssociation(map_landmarks_filtered, transformed_observations);

    double particle_weight = 1.0;

    // Find all associations and calculate the milti-variable gaussian probability.
    for(LandmarkObs map_landmark : map_landmarks_filtered) {
      for(LandmarkObs obs_landmark : transformed_observations) {
        // If found, multiply the probabilities. 
        if(map_landmark.id == obs_landmark.id) {
          double prob = multiv_prob(std_landmark[0], std_landmark[1], 
                                    obs_landmark.x, obs_landmark.y, 
                                    map_landmark.x, map_landmark.y);
          particle_weight *= prob;
          break;
        }
      }
    }

    weights.push_back(particle_weight);
    particle.weight = particle_weight;
  }
}

void ParticleFilter::resample() {

  vector<Particle> new_samples;

  // Pick samples based on the discrete distribution.
  std::discrete_distribution<int> disc_dist(weights.begin(), weights.end());
  
  for(int i = 0; i < num_particles; i++) {
    int index = disc_dist(gen);
    new_samples.push_back(particles[index]);
  }

  // Assign new samples to the particles vector
  particles = new_samples;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}