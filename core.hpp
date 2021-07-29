#pragma once

#include <algorithm>
#include <cassert>
#include <ostream>
#include <tuple>
#include <vector>

namespace asher {

typedef std::vector<std::vector<double>> Data;
typedef std::vector<double> RealV;


class DataView {
    public:

        typedef Data::iterator iterator;
        typedef Data::const_iterator const_iterator;
        
        DataView() = default;
        
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
            int n = size();
            return n>1? sum_sqdev(column)/(n-1) : 0;
        }
        
        double sum_sqdev(int column) const {
            double avg = mean(column);
            double result = 0;
            for (auto it = m_begin; it != m_end; ++it) {
                double dev = (*it)[column] - avg;
                result += dev*dev;
            }
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

}
