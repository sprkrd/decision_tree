
// Reference link: https://www.johndcook.com/blog/standard_deviation/
// I've actually improved on top of that version to also support the removal
// of data points (pop). Of course, the class has no way of knowing that the
// removed data point was, effectively, a point added in the past, so it relies
// on the caller knowing what it is doing.
//
// My version uses less members, as the old variance and average have to be
// remembered exclusively inside the update methods.

namespace asher {


class RunningStats {

    public:

        RunningStats() : m_avg(0), m_s(0), m_n(0) {
        }

        RunningStats(double avg, double var, int n) : m_avg(avg), m_s(var*(n-1)), m_n(n) {
        }

        void push(double x) {
            double prev_avg = m_avg;
            ++m_n;
            m_avg += 1.0/m_n * (x-m_avg);
            m_s += (x-m_avg)*(x-prev_avg);
        }

        void pop(double x) {
            double prev_avg = m_avg;
            --m_n;
            m_avg -= 1.0/m_n * (x-m_avg);
            m_s -= (x-m_avg)*(x-prev_avg);
        }

        double mean() const {
            return m_avg;
        }

        double variance() const {
            return m_n>1? m_s/(m_n-1) : 0;
        }

    private:

        double m_avg;
        double m_s;
        int m_n;

};


}

