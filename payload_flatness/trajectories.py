import numpy as np
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


def skew_matrix(x):
    a1 = x[0]
    a2 = x[1]
    a3 = x[2]
    A = np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]], dtype=np.double)
    return A

def ref_circular_trajectory(t, p, w_c, c):
    # Compute the desired Trajecotry of the system
    # COmpute Desired Positions
    cx = c[0]
    cy = c[1]
    cz = c[2]

    xd = cx + p * np.sin(w_c*t)
    yd = cy + p * np.cos(w_c*t)
    zd = cz + 0 * np.zeros((t.shape[0]))

    # Compute velocities
    xd_p =  p * w_c * np.cos(w_c * t)
    yd_p = -p * w_c * np.sin(w_c * t)
    zd_p = 0 * np.zeros((t.shape[0]))

    # Compute acceleration
    xd_pp = - p * w_c * w_c * np.sin(w_c * t)
    yd_pp = - p * w_c * w_c * np.cos(w_c * t) 
    zd_pp = 0 * np.zeros((t.shape[0]))

    # Compute jerk
    xd_ppp = - p * w_c * w_c * w_c * np.cos(w_c * t)
    yd_ppp =  p * w_c * w_c * w_c * np.sin(w_c * t) 
    zd_ppp = 0 * np.zeros((t.shape[0]))

    # Compute snap
    xd_pppp = p * w_c * w_c * w_c * w_c * np.sin(w_c * t)
    yd_pppp = p * w_c * w_c * w_c * w_c * np.cos(w_c * t)
    zd_pppp = 0 * np.zeros((t.shape[0]))

    # Compute angular displacement
    theta = 0 * np.zeros((t.shape[0]))

    # Compute angular velocity
    theta_p = 0 * np.zeros((t.shape[0]))
    #theta = np.arctan2(yd_p, xd_p)
    #theta = theta

    # Compute angular velocity
    #theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
    #theta_p[0] = 0.0

    theta_pp = 0 * np.zeros((theta.shape[0]))

    hd = np.vstack((xd, yd, zd))
    hd_p = np.vstack((xd_p, yd_p, zd_p))
    hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
    hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))
    hd_pppp = np.vstack((xd_pppp, yd_pppp, zd_pppp))
    return hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp

def compute_flatness_states(L, t, p, w_c, c):

    # Drone Parameters
    m = L[0]
    Jxx = L[1]
    Jyy = L[2]
    Jzz = L[3]
    g = L[4]
    Dxx = L[5]
    Dyy = L[6]
    Dzz = L[7]

    # Inertial Frame 
    Zw = np.array([[0.0], [0.0], [1.0]])
    Xw = np.array([[1.0], [0.0], [0.0]])
    Yw = np.array([[0.0], [1.0], [0.0]])

    hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp = ref_circular_trajectory(t, p, w_c, c)

    # Empty vector for the internal values
    alpha =  np.zeros((3, hd.shape[1]), dtype=np.double)
    beta =  np.zeros((3, hd.shape[1]), dtype=np.double)

    # Desired Orientation matrix only yaw
    Yc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zc = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Body Frame unit vectors
    Yb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zb = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Quaternions
    q = np.zeros((4, hd.shape[1]), dtype=np.double)

    # Force
    f = np.zeros((1, hd.shape[1]), dtype=np.double)

    # Angular vlocity
    w = np.zeros((3, hd.shape[1]), dtype=np.double)

    for k in range(0, hd.shape[1]):
        # Auxiliary variables
        alpha[:, k] = hd_pp[:, k] + g*Zw[:, 0] + (Dxx/m)*hd_p[:, k]
        beta[:, k] = hd_pp[:, k] + g*Zw[:, 0] + (Dyy/m)*hd_p[:, k]

        # Components Desired Orientation matrix
        Xc[:, k] = np.array([ np.cos(theta[k]), np.sin(theta[k]), 0])
        Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0])
        Zc[:, k] = np.array([0.0, 0.0, 1.0])

        # Body frame that is projected to the desired orientation
        Xb[:, k] = (np.cross(Yc[:, k], alpha[:, k]))/(np.linalg.norm(np.cross(Yc[:, k], alpha[:, k])))
        Yb[:, k] = (np.cross(beta[:, k], Xb[:, k]))/(np.linalg.norm(np.cross(beta[:, k], Xb[:, k])))
        Zb[:, k] = np.cross(Xb[:, k], Yb[:, k])

        R_d = np.array([[Xb[0, k], Yb[0, k], Zb[0, k]], [Xb[1, k], Yb[1, k], Zb[1, k]], [Xb[2, k], Yb[2, k], Zb[2, k]]])
        r_d = R.from_matrix(R_d)
        quad_d_aux = r_d.as_quat()
        q[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
        if k > 0:
            aux_dot = np.dot(q[:, k], q[:, k-1])
            if aux_dot < 0:
                q[:, k] = -q[:, k]
            else:
                q[:, k] = q[:, k]
        else:
            pass
        q[:, k] = q[:, k]/np.linalg.norm(q[:, k])
        # Compute nominal force of the in the body frame
        aux_z = m*hd_pp[:, k] + m*g*Zw[:, 0] + Dzz*hd_p[:, k]
        f[:, k] = np.dot(Zb[:, k], aux_z)
    return hd, hd_p, hd_pp, hd_ppp, hd_pppp, q, f, w

def compute_flatness_states_old(L, t, p, w_c, c):
    print("Fernando")
    # Drone Parameters
    m = L[0]
    Jxx = L[1]
    Jyy = L[2]
    Jzz = L[3]
    g = L[4]
    Dxx = L[5]
    Dyy = L[6]
    Dzz = L[7]

    # Inertial Frame 
    Zw = np.array([[0.0], [0.0], [1.0]])

    hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp = ref_circular_trajectory(t, p, w_c, c)


    # Desired Orientation matrix only yaw
    Yc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zc = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Body Frame unit vectors
    Yb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zb = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Quaternions
    q = np.zeros((4, hd.shape[1]), dtype=np.double)

    # Force
    f = np.zeros((1, hd.shape[1]), dtype=np.double)
    F = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Angular vlocity
    w = np.zeros((3, hd.shape[1]), dtype=np.double)

    for k in range(0, hd.shape[1]):
        # Auxiliary variables
        F[:, k] =  m*(hd_pp[:, k]+g*Zw[:, 0])

        norm_fb3 = np.linalg.norm(F[:, k])
        f[0, k] = norm_fb3 

        # Components Desired Orientation matrix
        Xc[:, k] = np.array([ np.cos(theta[k]), np.sin(theta[k]), 0])
        Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0])
        Zc[:, k] = np.array([0.0, 0.0, 1.0])

        # Body frame that is projected to the desired orientation
        Zb[:, k]  = F[:, k] / norm_fb3 ;
        Xb[:, k] = (np.cross(Yc[:, k], Zb[:, k]))/(np.linalg.norm(np.cross(Yc[:, k], Zb[:, k])))
        Yb[:, k] = np.cross(Zb[:, k], Xb[:, k])

        R_d = np.array([[Xb[0, k], Yb[0, k], Zb[0, k]], [Xb[1, k], Yb[1, k], Zb[1, k]], [Xb[2, k], Yb[2, k], Zb[2, k]]])
        r_d = R.from_matrix(R_d)
        quad_d_aux = r_d.as_quat()
        q[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
        if k > 0:
            aux_dot = np.dot(q[:, k], q[:, k-1])
            if aux_dot < 0:
                q[:, k] = -q[:, k]
            else:
                q[:, k] = q[:, k]
        else:
            pass
        q[:, k] = q[:, k]/np.linalg.norm(q[:, k])
        # Compute nominal force of the in the body frame
    return hd, hd_p, hd_pp, hd_ppp, hd_pppp, q, f, w