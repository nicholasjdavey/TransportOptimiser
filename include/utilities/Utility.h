#ifndef UTILITY_H
#define UTILITY_H

namespace Utility {
    /**
     * Converts the subscripts (rows,columns) used to address an Eigen matrix
     * (only 2D) to single-valued, column-major-based indices, similar to
     * Matlab's sub2ind function.
     *
     * @param X as m by n matrix (Eigen)
     * @param R list of row indices (Eigen)
     * @param C list of column indices (Eigen)
     * @return Y list of indices (Eigen)
     */
    template < typename DerivedX,
            typename DerivedR,
            typename DerivedC,
            typename DerivedY>
    static void sub2ind(
            const Eigen::PlainObjectBase<DerivedX>& X,
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

    /**
     * Converts the single-valued, column-major-based indices used to address
     * an Eigen matrix to subscripts(rows,columns) (only 2D), similar to
     * Matlab's ind2sub function.
     *
     * @param X as m by n matrix (Eigen)
     * @param Y list of indices (Eigen)
     * @return R list of row indices (Eigen)
     * @return C list of column indices (Eigen)
     */
    template < typename DerivedX,
            typename DerivedY,
            typename DerivedR,
            typename DerivedC>
    static void ind2sub(
            const Eigen::PlainObjectBase<DerivedX>& X,
            const Eigen::PlainObjectBase<DerivedY>& Y,
            Eigen::PlainObjectBase<DerivedR>& R,
            Eigen::PlainObjectBase<DerivedC>& C
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
}

#endif
