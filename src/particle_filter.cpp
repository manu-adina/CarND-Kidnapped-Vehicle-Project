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
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> angle_theta(theta, std[2]);

  for(int i = 0; i < num_particles; i++) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_x(gen);
    sample_theta = angle_theta(gen);
    Particle new_particle = {i, sample_x, sample_y, sample_theta, 1};
    particles.push_back(new_particle);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  double x_f, y_f, theta_f;
  
  for(Particle particle : particles) {
    x_f = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
    y_f = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - sin(particle.theta + yaw_rate * delta_t));
    theta_f = particle.theta + yaw_rate * delta_t;

    // Adding uncertainty
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x_f, std_pos[0]);
    std::normal_distribution<double> dist_y(y_f, std_pos[1]);
    std::normal_distribution<double> angle_theta(theta_f, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = angle_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  // Compare predicted to observations, and find min distance. Assign predicted.id to obs.id
  for(int i = 0; i < observations.size(); i++) {
    double min_dist = std::numeric_limits<double>::max();
    for(int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(distance < min_dist) {
        min_dist = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // For each particle update the weight.
  for(Particle particle : particles) {
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
      double map_landmark_x = map_landmarks.landmark_list[i].x_f;
      double map_landmark_y = map_landmarks.landmark_list[i].y_f;
      if(dist(particle.x, particle.y, map_landmark_x, map_landmark_y) <= sensor_range) {
        LandmarkObs landmark_in_range = {map_landmarks.landmark_list[i].id_i, map_landmark_x, map_landmark_x};
        map_landmarks_filtered.push_back(landmark_in_range);
      }
    }

    // Associate the observations to the landmarks.
    dataAssociation(map_landmarks_filtered, transformed_observations);

    bool found = false; // If there are no landmarks in range on the map.
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
          found = true;
          break;
        }
      }
    }

    // In case there are no landmarks, assign weight to 0.
    if(!found) particle_weight = 0;

    particle.weight = particle_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */



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