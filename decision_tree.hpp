#pragma once

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "core.hpp"
#include "running_stats.hpp"

namespace asher {

constexpr double DEF_MIN_IMPURITY_DECREASE = 0;
constexpr int DEF_MIN_SIZE_TO_SPLIT = 2;
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
            auto extended_data = extend_matrix(x, y);
            DataView data_view(extended_data);
            fit_aux(data_view, new_node(data_view));
        }

        double predict(const RealV& x) const {
            return predict_aux(x, 0);
        }

        RealV predict(const Data& x) const {
            RealV y(x.size());
            for (unsigned i = 0; i < x.size(); ++i) {
                y[i] = predict(x[i]);
            }
            return y;
        }

        std::string to_dot() const {
            std::ostringstream oss;
            oss << "digraph {\n";
            to_dot(oss, 0);
            oss << "}\n";
            return oss.str();
        }

    private:

        struct Node {
            double threshold;
            double mean;
            double variance;
            double split_gain;
            int sample_size;
            int feature_split;
            int left_child;
            int right_child;
        };

        std::tuple<double,double> optimal_split(Node& node, DataView& data, int feature) {
            data.sort(feature);
            double max_gain = -1;
            double best_threshold = 0;
            
            RunningStats stats_left;
            RunningStats stats_right(node.mean,
                                     node.variance*(data.size()-1),
                                     data.size());

            double var_before_split = node.variance;
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
                }
                
            }
            return std::make_tuple(best_threshold, max_gain);
        }

        std::tuple<int,double,double> optimal_split(Node& node, DataView& data) {
            int best_feature = 0;
            double best_threshold = 0;
            double max_gain = -1;
            for (int feature = 0; feature < m_feat_count; ++feature) {
                auto[threshold,gain] = optimal_split(node, data, feature);
                if (gain > max_gain) {
                    best_feature = feature;
                    best_threshold = threshold;
                    max_gain = gain;
                }
            }
             
            return std::make_tuple(best_feature, best_threshold, max_gain);
        }
        
        int new_node(const DataView& data) {
            Node& node = m_tree.emplace_back();
            node.threshold = node.split_gain = 0;
            node.mean = data.mean(m_feat_count);
            node.variance = data.variance(m_feat_count);
            node.sample_size = data.size();
            node.feature_split = node.left_child = node.right_child = -1;
            return m_tree.size()-1;
        }

        void fit_aux(DataView& data, int node_index, int depth = 0) {
            int min_size_to_split = std::max(m_parameters.min_size_to_split, 2*m_parameters.min_leaf_size);
            if (depth >= m_parameters.max_depth || data.size() < min_size_to_split) {
                return;
            }
            
            auto[feature,threshold,gain] = optimal_split(m_tree[node_index], data);
        
            if (gain < m_parameters.min_impurity_decrease) {
                return;
            }
            auto[left,right] = data.partition(feature, threshold);
              
            int left_child = new_node(left);
            int right_child = new_node(right);

            Node& node = m_tree[node_index];
            
            node.threshold = threshold;
            node.split_gain = gain;
            node.feature_split = feature;
            node.left_child = left_child;
            node.right_child = right_child;;
 
            fit_aux(left, left_child, depth+1);
            fit_aux(right, right_child, depth+1);
        }

        double predict_aux(const RealV& x, int node_index) const {
            const Node& node = m_tree[node_index];
            if (node.feature_split == -1) {
                return node.mean;
            }
            int child = x[node.feature_split]<=node.threshold? node.left_child : node.right_child;
            return predict_aux(x, child);
        }

        void to_dot(std::ostringstream& oss, int node_index) const {
            const Node& node = m_tree[node_index];

            const char* shape = node.feature_split == -1? "box" : "ellipse";

            oss << node_index << " [shape=" << shape << ",label=<";

            oss << "<b>mean:</b> " << node.mean << "<br/>"
                << "<b>var:</b> " << node.variance << "<br/>"
                << "<b>N:</b> " << node.sample_size;

            if (node.feature_split != -1) {
                oss << "<br/>"
                    << "<b>imp. reduc.:</b> " << node.split_gain << "<br/>"
                    << "<b>split:</b> x[" << node.feature_split << "] &#8804; " << node.threshold;
            }
            oss << ">];\n";

            if (node.feature_split != -1) {
                oss << node_index << " -> " << node.left_child << '\n'
                    << node_index << " -> " << node.right_child << '\n';

                to_dot(oss, node.left_child);
                to_dot(oss, node.right_child);
            }
        }
        
        Parameters m_parameters;
        std::vector<Node> m_tree;
        int m_feat_count;

};


}




