/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// Set  the number of particles to draw
  num_particles = 128;

  // noise generation
  default_random_engine gen;
  normal_distribution<double> Normal_x_initialize(0, std[0]);
  normal_distribution<double> Normal_y_initialize(0, std[1]);
  normal_distribution<double> Normal_theta_initialize(0, std[2]);

  // add particles at iniital position + noise with weight 1.0 for all
  for (int i = 0; i < num_particles; i++) {
    Particle particle_tmp;
    particle_tmp.id = i;
    particle_tmp.x = x + Normal_x_initialize(gen);
    particle_tmp.y = y + Normal_y_initialize(gen);
    particle_tmp.theta = theta + Normal_theta_initialize(gen);
    particle_tmp.weight = 1.0;
    weights.push_back(particle_tmp.weight);
    particles.push_back(particle_tmp);
  }

  // Change the flag to inidicate it is initialized now
  is_initialized = true;
}

// Calculates the bivariate normal pdf of a point given a mean and std and assuming zero correlation
inline double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
  return exp(-((x - mu_x)*(x - mu_x) / (2 * sig_x*sig_x) + (y - mu_y)*(y - mu_y) / (2 * sig_y*sig_y))) / (2 * M_PI*sig_x*sig_y);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// noise generation using normal distribution
  default_random_engine gen;
  normal_distribution<double> Normal_x_initialize(0, std_pos[0]);
  normal_distribution<double> Normal_y_initialize(0, std_pos[1]);
  normal_distribution<double> Normal_theta_initialize(0, std_pos[2]);

  for (unsigned int i=0; i < particles.size(); i++) {

    // define temporary variables
    double x_new;
    double y_new;
    double theta_new;

    // update next state: if yaw_rate too small, not using ctrm.
		// calculate new state
    if (fabs(yaw_rate) < 0.0001) {
      x_new = particles[i].x + velocity*cos(particles[i].theta)*delta_t;
      y_new = particles[i].y + velocity*sin(particles[i].theta)*delta_t;
      theta_new = particles[i].theta + yaw_rate*delta_t;
    } else {
      x_new = particles[i].x + velocity / yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      y_new = particles[i].y + velocity / yaw_rate*(-cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta));
      theta_new = particles[i].theta + yaw_rate*delta_t;
    }

    // update particle adding noises
    particles[i].x = x_new + Normal_x_initialize(gen);
    particles[i].y = y_new + Normal_y_initialize(gen);
    particles[i].theta = theta_new + Normal_theta_initialize(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// iterate through observations
  for (unsigned i = 0; i < observations.size(); i++) {

    // initialize temporary distance and closest id
    double cur_range = 1e6;
    int closest_j = -1;

    // search through landmarks to find the closest one
    for (unsigned j = 0; j < predicted.size(); ++j) {

      // calculate Euclidian distance between two points
      double eval_range = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // assign the minium range found so far
      if (eval_range < cur_range) {
        cur_range = eval_range;
        closest_j = j;
      }
    }
    // assign the observation the closest id
    observations[i].id = predicted[closest_j].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	// empty all particles' weights
	weights.clear();

	// search through all particles
	for (unsigned i = 0; i < particles.size(); i++) {

		// Transform the coordinates of the observations to map coordinates from vehicle view
		std::vector<LandmarkObs> obs_map_coord;
		for (unsigned int j = 0; j < observations.size(); ++j) {
			if (dist(observations[j].x, observations[j].y, 0, 0) <= sensor_range) {
				LandmarkObs obs_tmp;
				obs_tmp.x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
				obs_tmp.y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);
				obs_tmp.id = -1;
				obs_map_coord.push_back(obs_tmp);
			}
		}


		// generate all reachable landmarks in map coordintes
		std::vector<LandmarkObs> close_landmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
				LandmarkObs obs_tmp;
				obs_tmp.x = map_landmarks.landmark_list[j].x_f;
				obs_tmp.y = map_landmarks.landmark_list[j].y_f;
				obs_tmp.id = map_landmarks.landmark_list[j].id_i;
				close_landmarks.push_back(obs_tmp);
			}
		}


		// data association by finding the closest landmark id for each observaton
		dataAssociation(close_landmarks, obs_map_coord);

		// calculate the weights by multiplying the probability values
		double weight = 1;
		for (unsigned int j = 0; j < close_landmarks.size(); j++) {
			double min_range = 1e6;
			int min_lm = -1;
			// find the map coordinates for the landmark
			for (unsigned int k = 0; k < obs_map_coord.size(); ++k) {
				// find the landmark id with minimum range to the sample
				if (obs_map_coord[k].id == close_landmarks[j].id) {
					double eval_range = dist(close_landmarks[j].x, close_landmarks[j].y, obs_map_coord[k].x, obs_map_coord[k].y);
					if (eval_range < min_range) {
						min_range = eval_range;
						min_lm = k;
					}
				}
			}

			if (min_lm != -1) {
				weight *= bivariate_normal(obs_map_coord[min_lm].x, obs_map_coord[min_lm].y, close_landmarks[j].x, close_landmarks[j].y, std_landmark[0], std_landmark[1]);
			}
		}

		// update the weight of the particle
		weights.push_back(weight);
		particles[i].weight = weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resample_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
	default_random_engine gen;
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resample_particles.push_back(particles[index]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
