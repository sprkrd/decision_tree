#pragma once

#include <iostream>
#include <limits>

#include "core.hpp"
#include "running_stats.hpp"

namespace asher {

constexpr double DEF_MIN_IMPURITY_DECREASE = 0;
constexpr int DEF_MIN_SIZE_TO_SPLIT = 5;
constexpr int DEF_MIN_LEAF_SIZE = 1;
constexpr int DEF_MAX_DEPTH = std::numeric_limits<int>::max();

class DecisionTreeRegressor {

    public:
        struct Parameters {
            double min_impurity_decrease;
            int min_size_to_split;
            int min_leaf_size;
            int max_depth;
        };
        
        DecisionTreeRegressor() {
            m_parameters.min_impurity_decrease = DEF_MIN_IMPURITY_DECREASE;
            m_parameters.min_size_to_split = DEF_MIN_SIZE_TO_SPLIT;
            m_parameters.min_leaf_size = DEF_MIN_LEAF_SIZE;
            m_parameters.max_depth = DEF_MAX_DEPTH;
        }
        
        DecisionTreeRegressor(const Parameters& parameters) : m_parameters(parameters) {
        }
        
        const Parameters& get_parameters() const {
            return m_parameters;
        }
        
        DecisionTreeRegressor& set_min_impurity_decrease(double min_impurity_decrease) {
            m_parameters.min_impurity_decrease = min_impurity_decrease;
            return *this;
        }
        
        DecisionTreeRegressor& set_min_size_to_split(int min_size_to_split) {
            m_parameters.min_size_to_split = min_size_to_split;
            return *this;
        }
        
        DecisionTreeRegressor& set_min_leaf_size(int min_leaf_size) {
            m_parameters.min_leaf_size = min_leaf_size;
            return *this;
        }
        
        DecisionTreeRegressor& set_max_depth(int max_depth) {
            m_parameters.max_depth = max_depth;
            return *this;
        }
        

        void fit(const Data& x, const RealV& y) {
            assert(x.size() > 0 && x[0].size() > 0 && "no data has been given");
            m_tree.clear();
            m_feat_count = x[0].size();
            m_data = extend_matrix(x, y);
            int root = new_node(DataView(m_data));
            fit_aux(m_tree[root], 1);
        }

        double predict(const RealV& x) const;

        RealV predict(const Data& x) const;

    private:
        struct Node {
            DataView data;
            double threshold;
            double mean;
            double variance;
            double split_gain;
            int feature_split;
            int left_child;
            int right_child;
            
            bool is_leaf() const {
                return feature_split < 0;
            }
        };

        std::tuple<double,double> optimal_split(DataView data, int feature) {
            data.sort(feature);
            double max_gain = -1;
            double best_threshold = 0;
            
            RunningStats stats_left;
            RunningStats stats_right(data.mean(m_feat_count),
                                     data.sum_sqdev(m_feat_count),
                                     data.size());

            double var_before_split = stats_right.variance();
            int i = 0;
            while (i < data.size()) {
                double x_i = data[i][feature];
                do {
                    double y = data[i].back();
                    stats_left.push(y);
                    stats_right.pop(y);
                    ++i;
                } while (i < data.size() && data[i][feature] == x_i);
                
                if (stats_left.size() < m_parameters.min_leaf_size) {
                    continue;
                }
                else if (stats_right.size() < m_parameters.min_leaf_size) {
                    break;
                }
                
                double avg_variance = (double)stats_left.size()/data.size() * stats_left.variance() +
                                      (double)stats_right.size()/data.size() * stats_right.variance();
                double gain = var_before_split - avg_variance;
                
                if (gain > max_gain) {
                    max_gain = gain;
                    best_threshold = i<data.size()? (x_i + data[i][feature])/2 : x_i;
                    std::cout << ">>> " << stats_left.size() << ' ' << stats_right.size() << ' ' << best_threshold << ' ' << i << std::endl;
                }
                
            }
            return std::make_tuple(best_threshold, max_gain);
        }

        std::tuple<int,double,double> optimal_split(DataView data) {
            int best_feature = 0;
            double best_threshold = 0;
            double max_gain = -1;
            for (int feature = 0; feature < m_feat_count; ++feature) {
                auto[threshold,gain] = optimal_split(data, feature);
                if (gain > max_gain) {
                    best_feature = feature;
                    best_threshold = threshold;
                    max_gain = gain;
                }
            }
            
            
            
            return std::make_tuple(best_feature, best_threshold, max_gain);
        }
        
        int new_node(DataView data) {
            //std::cout << __PRETTY_FUNCTION__ << std::endl;
            Node& node = m_tree.emplace_back();
            node.data = data;
            node.threshold = node.split_gain = 0;
            node.mean = data.mean(m_feat_count);
            node.variance = data.variance(m_feat_count);
            node.feature_split = node.left_child = node.right_child = -1;
            return m_tree.size()-1;
        }

        void fit_aux(Node& node, int depth = 1) {
            int min_size_to_split = std::max(m_parameters.min_size_to_split, 2*m_parameters.min_leaf_size);
            if (depth >= m_parameters.max_depth || node.data.size() < min_size_to_split) {
                return;
            }
            
            auto[feature,threshold,gain] = optimal_split(node.data);
        
            if (gain < m_parameters.min_impurity_decrease) {
                return;
            }
            
            auto[left,right] = node.data.partition(feature, threshold);
            
            if (left.size() < 5 || right.size() < 5) {
                std::cout << left << std::endl << std::endl;
                std::cout << right << std::endl << std::endl;
            }
            
            std::cout << feature << ' ' << threshold << ' ' << gain << ' ' << left.size() << ' ' << right.size() << std::endl;
            
            node.threshold = threshold;
            node.split_gain = gain;
            node.feature_split = feature;
            node.left_child = new_node(left);
            node.right_child = new_node(right);
            
            fit_aux(m_tree[node.left_child], depth+1);
            fit_aux(m_tree[node.right_child], depth+1);
        }
        
        Parameters m_parameters;
        Data m_data;
        std::vector<Node> m_tree;
        int m_feat_count;

};


}




