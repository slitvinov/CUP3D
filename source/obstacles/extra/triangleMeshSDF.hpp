#include <cmath>
#include <vector>

#ifndef CubismUP_3D_triangleMeshSDF_h
#define CubismUP_3D_triangleMeshSDF_h

template <typename T>
struct Vector3 {
    T &operator[](int k) { return x_[k]; }
    const T &operator[](int k) const { return x_[k]; }

    friend Vector3 operator+(const Vector3 &a, const Vector3 &b) {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }

    friend Vector3 operator-(const Vector3 &a, const Vector3 &b) {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }

    friend Vector3 operator*(const T &a, const Vector3 &b) {
        return {a*b[0], a*b[1], a*b[2]};
    }

    friend Vector3 cross(const Vector3 &a, const Vector3 &b) {
        return Vector3{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    }

    friend auto dot(Vector3 a, Vector3 b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    friend auto norm(Vector3 &a) {
        return std::sqrt( dot(a,a) );
    }

    T x_[3];
};

// From https://stackoverflow.com/a/253874
inline bool approximatelyEqual( Real a, Real b ) {
    auto epsilon = std::numeric_limits<Real>::epsilon();
    return std::abs(a - b) <= std::max(std::abs(a), std::abs(b)) * epsilon;
};

inline Real pointTriangleSqrDistance(
        const Vector3<Vector3<Real>> &tri,
        const Vector3<Real> p) {
    const auto n = cross(tri[1] - tri[0], tri[2] - tri[0]);
    const auto p0 = p - tri[0];
    const auto p1 = p - tri[1];
    const auto p2 = p - tri[2];
    const auto n0 = cross(p1, p2);
    const auto n1 = cross(p2, p0);
    const auto n2 = cross(p0, p1);
    const bool inside0 = dot(n0, n) >= 0;
    const bool inside1 = dot(n1, n) >= 0;
    const bool inside2 = dot(n2, n) >= 0;
    if (inside0 && inside1 && inside2) {
        auto dot_ = dot(p - tri[0], n);
        return dot_ * dot_ / dot(n, n);
    }
    Vector3<Real> pu;
    Vector3<Real> vu;
    if (!inside0){
        vu = tri[2] - tri[1];
        pu = p1;
    } else if (!inside1) {
        vu = tri[0] - tri[2];
        pu = p2;
    } else {
        vu = tri[1] - tri[0];
        pu = p0;
    }
    const auto vuvu = dot(vu, vu);
    const auto vupu = dot(vu, pu);
    if (vupu <= 0) {
        return dot(pu, pu);
    } else if (vupu >= vuvu) {
        const auto pv = pu - vu;
        return dot(pv, pv);
    } else {
        const auto out = dot(pu, pu) - vupu * vupu / vuvu;
        return out >= 0 ? out : 0;
    }
}

class Mesh {
public:
    Mesh(const std::vector<Vector3<Real>> &x,
         const std::vector<Vector3<int>>  &tri) :
        x_{x}, tri_{tri}
    { }

    void rotate( const Real Rmatrix[3][3], const double position[3]) {
        for( auto& pt: x_ ){
            // rotate point
            pt = {
              Rmatrix[0][0]*pt[0] + Rmatrix[1][0]*pt[1] + Rmatrix[2][0]*pt[2],
              Rmatrix[0][1]*pt[0] + Rmatrix[1][1]*pt[1] + Rmatrix[2][1]*pt[2],
              Rmatrix[0][2]*pt[0] + Rmatrix[1][2]*pt[1] + Rmatrix[2][2]*pt[2]
            };
            // translate point
            pt = { pt[0]+position[0], pt[1]+position[1], pt[2]+position[2] };
        }
    }

    Real nonConvexSDF(Vector3<Real> p, Real h) const {
        // Find the closest triangles and the distance to them.
        std::vector<Vector3<Vector3<Real>>> closest{};
        Real minSqrDist = 1e100;
        for (int i = 0; i < (int)tri_.size(); ++i) {
            Vector3<Vector3<Real>> t{
                x_[tri_[i][0]],
                x_[tri_[i][1]],
                x_[tri_[i][2]],
            };
            const Real sqrDist = pointTriangleSqrDistance(t, p);
            if( approximatelyEqual( sqrDist, minSqrDist ) )
                closest.push_back(t);
            else if (sqrDist < minSqrDist) {
                minSqrDist = sqrDist;
                closest.clear();
                closest.push_back(t);
            }
        }
        const auto dist = std::sqrt(minSqrDist) > 2*h ? 2*h : std::sqrt(minSqrDist);

        // Check on which side of the closest triangle we are. Compute normal
        Vector3<Real> n{};
        if( closest.size() == 1 )
            n = cross(closest[0][1] - closest[0][0], closest[0][2] - closest[0][0]);     
        else if( closest.size() == 2 ){
            // Closest point is on an edge, average the normals
            auto n1 = cross(closest[0][1] - closest[0][0], closest[0][2] - closest[0][0]);
            auto n2 = cross(closest[1][1] - closest[1][0], closest[1][2] - closest[1][0]);
            n = n1 + n2;
        }
        else { 
            // Closest point is on a vertex, angle-weighted average (http://www2.compute.dtu.dk/pubdb/pubs/1833-full.html)
            for( const auto& triangle : closest ){
                size_t i = 0;
                for( ; i < 3; i++ )
                {
                    auto dir = p - triangle[i];
                    if( approximatelyEqual( minSqrDist, dot(dir, dir) ) )
                        break;
                }
                // compute angle at edge of triangle 
                auto edgeVec1 = triangle[ (i+1)%3 ] - triangle[i];
                auto edgeVec2 = triangle[ (i+2)%3 ] - triangle[i];
                Real cosalphai = dot(edgeVec1, edgeVec2) / (norm(edgeVec1)*norm(edgeVec2));
                Real alphai  = std::acos(cosalphai);
                // compute normal of triangle
                auto ni = cross(triangle[1] - triangle[0], triangle[2] - triangle[0]);
                // normalize
                ni = ( 1 / norm(ni) ) * ni;
                // angle weighted sum of norm
                n = n + alphai*ni;
            }
        }
        const auto side = dot(n, p - closest[0][0]);
        return std::copysign(dist, side); 
    }

    std::vector<Vector3<Real>> x_;
    std::vector<Vector3<int>> tri_;
};

#endif // CubismUP_3D_triangleMeshSDF_h
