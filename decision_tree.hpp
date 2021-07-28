#include <algorithm>
#include <cassert>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace asher {

typedef std::vector<std::vector<double>> Data;
typedef std::vector<double> RealV;


class DataView {
    public:

        typedef Data::iterator iterator;
        typedef Data::const_iterator const_iterator;
        
        DataView(Data& data) : DataView(data.begin(), data.end()) {
        }

        DataView(iterator begin, iterator end) : m_begin(begin), m_end(end) {
        }

        void sort(int feature) {
            std::sort(m_begin, m_end,
                [=](const RealV& l, const RealV& r) {
                    return l[feature] < r[feature];
                });
        }

        std::tuple<DataView,DataView> partition(int feature, double threshold) {
            iterator left = m_begin;
            iterator right = m_end-1;
            while (true) {
                while (left <= right && (*left)[feature] <= threshold) {
                    ++left;
                }
                while (left <= right && (*right)[feature] > threshold) {
                    --right;
                }
                if (left < right) {
                    std::swap(*left, *right);
                    ++left;
                    --right;
                }
                else {
                    break;
                }
            }
            return {DataView(m_begin, left), DataView(left, m_end)};
        }
        
        iterator begin() {
            return m_begin;
        }

        iterator end() {
            return m_end;
        }

        const_iterator begin() const {
            return m_begin;
        }

        const_iterator end() const {
            return m_end;
        }

        RealV& operator[](int index) {
            return *(m_begin+index);
        }

        double mean(int column) const {
            double result = 0;
            for (auto it = m_begin; it != m_end; ++it) {
                result += (*it)[column];
            }
            result /= size();
            return result;
        }

        double variance(int column) const {
            double avg = mean(column);
            double result = 0;
            for (auto it = m_begin; it != m_end; ++it) {
                double dev = (*it)[column] - avg;
                result += dev*dev;
            }
            result /= size();
            return result;
        }

        const RealV& operator[](int index) const {
            return *(m_begin+index);
        }

        int size() const {
            return std::distance(m_begin, m_end);
        }

    private:

        iterator m_begin, m_end;
};

std::ostream& operator<<(std::ostream& out, const DataView& data) {
    bool first_row = true;
    if (!data.size()) {
        out << "(empty)";
    }
    for (const auto& row : data) {
        if (!first_row) {
            out << '\n';
        }
        first_row = false;
        bool first_col = true;
        for (double x : row) {
            if (!first_col) {
                out << ',';
            }
            first_col = false;
            out << x;
        }
    }
    return out;
}

Data extend_matrix(const Data& x, const RealV& y) {
    assert(x.size() == y.size() && "size mistmatch between x and y");
    Data x_y = x;
    for (unsigned i = 0; i < x.size(); ++i) {
        x_y[i].push_back(y[i]);
    }
    return x_y;
}

double variance(double sum, double sum_sq, int n) {
    return (sum_sq - sum*sum/n)/n;
}


class DecisionTreeRegressor {

    public:



        void fit(const Data& x, const RealV& y) {
            auto xy = extend_matrix(x, y);
            DataView view(xy.begin(), xy.end());

        }

        double predict(const RealV& x) const;

        RealV predict(const Data& x) const;

    private:

        std::tuple<double,double> optimal_split(DataView& view, int feature) {
            view.sort(feature);
            double max_gain = 0;
            double best_threshold = 0;

            double sum_left = 0;
            double sum_right = 0;
            double sum_sq_left = 0;
            double sum_sq_right = 0;
            for (const auto& row : view) {
                double y = row.back();
                sum_right += y;
                sum_sq_right += y*y;
            }
            double var_before_split = variance(sum_right, sum_sq_right, view.size());
            int i = 0;
            while (i < view.size()) {
                double x_i = view[i][feature];
                while (true) {
                    double y = view[i].back();
                    sum_left += y;
                    sum_right -= y;
                    sum_sq_left += y*y;
                    sum_sq_right -= y*y;
                    if (i+1 < view.size() && view[i+1][feature] == x_i) {
                        ++i;
                    }
                    else {
                        break;
                    }
                }
                int size_left = i+1;
                int size_right = view.size() - size_left;
                double variance_left = variance(sum_left, sum_sq_left, size_left);
                double variance_right = variance(sum_right, sum_sq_right, size_right);
                double avg_variance = (double)size_left/view.size()*variance_left +
                                      (double)size_right/view.size()*variance_right;
                double gain = var_before_split - avg_variance;

                if (gain > max_gain) {
                    max_gain = gain;
                    best_threshold = i+1<view.size()? (x_i + view[i+1][feature])/2 : x_i;
                }
                ++i;
            }
            return std::make_tuple(best_threshold, max_gain);
        }

        std::tuple<int,double,double> optimal_split(DataView& view) {
            int best_feature = 0;
            double best_threshold = 0;
            double max_gain = 0;
            for (int feature = 0; feature < (int)view[0].size()-1; ++feature) {
                auto[threshold,gain] = optimal_split(view, feature);
                if (gain > max_gain) {
                    best_feature = feature;
                    best_threshold = threshold;
                    max_gain = gain;
                }
            }
            return std::make_tuple(best_feature, best_threshold, max_gain);
        }

        //void fit_aux(DataView& view) {
            
        //}

        struct Node {
            DataView data;
            double threshold;
            double mean;
            double variance;
            double split_gain;
            int feature_split;
            int left_child;
            int right_child;
        };

        Data m_data;
        std::vector<Node> m_tree;

};


}




