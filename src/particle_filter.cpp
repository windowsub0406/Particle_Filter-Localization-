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

default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_psi(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	for (int i = 0; i < num_particles; i++) {

		double x_, y_, theta_;
		if (fabs(yaw_rate) > 0.001) {
			theta_ = particles[i].theta + yaw_rate * delta_t;
			x_ = particles[i].x + (velocity / yaw_rate) * (sin(theta_) - sin(particles[i].theta));
			y_ = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(theta_));
		}
		else {
			theta_ = particles[i].theta;
			x_ = particles[i].x + velocity * delta_t * cos(theta_);
			y_ = particles[i].y + velocity * delta_t * sin(theta_);
		}

		normal_distribution<double> dist_x(x_, std_pos[0]);
		normal_distribution<double> dist_y(y_, std_pos[1]);
		normal_distribution<double> dist_theta(theta_, std_pos[2]);

		// add noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for (int i = 0; i < observations.size(); i++) {
		double min_dist = 9999;
		for (int j = 0; j < predicted.size(); j++) {
			double dist_ = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (dist_ < min_dist) {
				min_dist = dist_;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
	double norm = 2 * M_PI * std_x * std_y;    
	double total_weight = 0.0;
	
    for (int i = 0; i < num_particles; i++) {

        double p_x	   = particles[i].x;
        double p_y	   = particles[i].y;
        double p_theta = particles[i].theta;
			
		vector<LandmarkObs> obs_map_indices;
		// Translate to map coordinates
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs obs_map;
			obs_map.x = (observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta)) + p_x;
			obs_map.y = (observations[j].x * sin(p_theta) + observations[j].y * cos(p_theta)) + p_y;
			obs_map_indices.push_back(obs_map);
		}
		
		vector<LandmarkObs> predicted_landmarks;
		// search landmarks within range
        for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
			double dist_ = dist(p_x, p_y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
			if (dist_ < sensor_range) {
				LandmarkObs predicted_lm;
				predicted_lm.id = map_landmarks.landmark_list[k].id_i;
				predicted_lm.x = map_landmarks.landmark_list[k].x_f;
				predicted_lm.y = map_landmarks.landmark_list[k].y_f;

				predicted_landmarks.push_back(predicted_lm);
			}
		}

        // find close landmarks
		dataAssociation(predicted_landmarks, obs_map_indices);

        double prob = 1.0;
        double x_, y_;
		particles[i].weight = 1.0;
		for (int j = 0; j < obs_map_indices.size(); j++) {

			for (int k = 0; k < predicted_landmarks.size(); k++) {

				// find corresponding landmark
				if (obs_map_indices[j].id == predicted_landmarks[k].id) {
					x_ = predicted_landmarks[k].x;
					y_ = predicted_landmarks[k].y;
					break;
				}
			}

			double x_diff = pow((obs_map_indices[j].x - x_) / std_x, 2.0);
			double y_diff = pow((obs_map_indices[j].y - y_) / std_y, 2.0);

			prob *= exp(-(x_diff + y_diff) / 2.0) / norm;
		}

        particles[i].weight = prob;
		weights[i] = prob;
		total_weight += prob;
    }

	// normalize by dividing by total weight
	if (total_weight != 0) {
		for (int i = 0; i < weights.size(); i++) {
			particles[i].weight /= total_weight;
			weights[i] /= total_weight;
		}
	}
}

void ParticleFilter::resample() {
	
	discrete_distribution<int> distrib(weights.begin(), weights.end());
	vector<Particle> update_particles;
	
	for (int i = 0; i < num_particles; i++) {
		update_particles.push_back(particles[distrib(gen)]);
	}
	particles = update_particles;
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
