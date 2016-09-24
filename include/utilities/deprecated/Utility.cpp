#include "../transportbase.h"

template < typename DerivedX,
        typename DerivedR,
        typename DerivedC,
        typename DerivedY>
void Utility::sub2ind(const Eigen::PlainObjectBase<DerivedX>& X,
        const Eigen::PlainObjectBase<DerivedR>& R,
        const Eigen::PlainObjectBase<DerivedC>& C,
        Eigen::PlainObjectBase<DerivedY>& Y) {

    if (R.size() != C.size()) {
        throw "Utility::sub2ind: Row and column index vectors must be the same length!";
    }

    if (((R.array()<0).sum() > 0) || ((C.array()<0).sum() > 0)) {
        throw "Utility::sub2ind: Matrix subscripts must be >= 0 !";
    }

    int xm = X.rows();
    int xn = X.cols();

    if (((R.array()>=xm).sum() > 0) || ((C.array()>=xn).sum() > 0)) {
        throw "Utility::sub2ind: Index vectors exceed matrix dimensions!";
    }

    Y = R + C*xm;
}

template < typename DerivedX,
        typename DerivedR,
        typename DerivedC,
        typename DerivedY>
void Utility::ind2sub(const Eigen::PlainObjectBase<DerivedX> &X,
        const Eigen::PlainObjectBase<DerivedY> &Y,
        Eigen::PlainObjectBase<DerivedR> &R,
        Eigen::PlainObjectBase<DerivedC> &C
        ) {

    int xm = X.rows();
    int xn = X.cols();

    if ((Y.array() >= xm*xn).sum() > 0) {
        throw "Index vector exceeds matrix size!";
    }

    Eigen::VectorXd tempvec = Y/xm;
    igl::floor(tempvec, C);
    R = Y - C*xm;
}

#ifdef UTILITY_STATIC_LIBRARY
// Explicit template specialization
#endif
