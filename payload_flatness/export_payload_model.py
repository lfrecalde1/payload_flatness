from acados_template import AcadosModel
from casadi import MX, MX, external, vertcat
from ctypes import *
import copy
import casadi as cs
import numpy as  np
import os
from acados_template import AcadosSim, AcadosSimSolver
from casadi import tanh, SX, Sparsity, hessian, if_else, horzcat, DM, blockcat
from casadi import Function
from casadi import jacobian
from payload_flatness import rotation_casadi, rotation_inverse_casadi, quaternion_multiplication_casadi

rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()
quat_multi = quaternion_multiplication_casadi()

EPS = 1e-4

def quatTorot_c(quat):
    # Normalized quaternion
    q = quat
    #q = q/(q.T@q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Q = cs.vertcat(
        cs.horzcat(q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)),
        cs.horzcat(2*(q1*q2+q0*q3), q0**2+q2**2-q1**2-q3**2, 2*(q2*q3-q0*q1)),
        cs.horzcat(2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2+q3**2-q1**2-q2**2))

    return Q

def rotation_matrix_error_norm_c():
    # Desired Quaternion
    qd = cs.MX.sym('qd', 4, 1)

    # Current quaternion
    q = cs.MX.sym('q', 4, 1)

    Rd = quatTorot_c(qd)
    R = quatTorot_c(q)

    error_matrix = (Rd.T@R - R.T@Rd)/2

    vector_error = cs.vertcat(error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0])

    error_orientation_scalar_f = Function('error_orientation_scalar_f', [qd, q], [vector_error])

    return error_orientation_scalar_f

def rotation_matrix_error_c():
    # Desired Quaternion
    qd = cs.MX.sym('qd', 4, 1)

    # Current quaternion
    q = cs.MX.sym('q', 4, 1)

    Rd = quatTorot_c(qd)
    R = quatTorot_c(q)

    error_matrix = R.T@Rd
    error_matrix_f = Function('error_matrix_f', [qd, q], [error_matrix])

    return error_matrix_f

## Functions from casadi 
rotation_error_norm_f = rotation_matrix_error_norm_c()
rotation_error_f = rotation_matrix_error_c()

def evaluate_quat():
    quat = MX.sym('quat', 4)
    mat = as_matrix(quat)
    f = cs.Function('f', [quat], [mat])
    result = f([0, 0, 0, 1])

def evaluate_var(name, var,
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        xi_x, xi_y, xi_z,
        xidot_x, xidot_y, xidot_z,
        u1, u2, u3, u4):

    f = cs.Function('f', [
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        xi_x, xi_y, xi_z,
        xidot_x, xidot_y, xidot_z,
        u1, u2, u3, u4],
        [var])

    hover = 4929
    hover = 0
    print(f"{name}: ")
    print(f(
        1, 0, 0 , 0,
        0, 0, 0,
        0, 0, -1,
        0, 0, 0,
        hover, hover, hover, hover))


def evaluate_var_final(name, var,
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        xi_x, xi_y, xi_z,
        xidot_x, xidot_y, xidot_z,
        u1, u2, u3, u4,
        xLdot, yLdot, zLdot,
        xdot, ydot, zdot):

    print("evaluating")
    f = cs.Function('f', [
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        xi_x, xi_y, xi_z,
        xidot_x, xidot_y, xidot_z,
        u1, u2, u3, u4,
        xLdot, yLdot, zLdot,
        xdot, ydot, zdot],
        [var])

    hover = 4.929
    hover = 4000
    print(f"{name}: ")
    print(f(
        1, 0, 0 , 0,
        0, 0, 0,
        0, 0, -1,
        0, 0, 0,
        hover, hover, hover, hover,
        0, 0, 0.0,
        0, 0, 0))

def evaluate_var_cross(name, var,
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        u1, u2, u3, u4,
        x, y, z,
        xdot, ydot, zdot,
        xL, yL, zL,
        xLdot, yLdot, zLdot,
        u1dot, u2dot, u3dot, u4dot):

    f = cs.Function('f', [
        qw, qx, qy, qz,
        omega_x, omega_y, omega_z,
        u1, u2, u3, u4,
        x, y, z,
        xdot, ydot, zdot,
        xL, yL, zL,
        xLdot, yLdot, zLdot,
        u1dot, u2dot, u3dot, u4dot],
        [var])

    hover = 139
    print(f"{name}: ")
    # print(f(
    #     1, 0, 0 , 0,
    #     0, 0, 0,
    #     175.50, 175.50, 175.50, 175.50,
    #     0, 0, 0.5,
    #     0 ,0 , 0,
    #     0, 0, 0.2,
    #     0, 0, 0,
    #     0, 0, 0, 0))

    print(f(
        1, 0, 0 , 0,
        0, 0, 0,
        136.83, 136.83, 136.83, 136.83,
        0, 0, 0.5,
        0 ,0 , 0.2,
        0, 0, 0.0,
        0, 0, 0.2,
        0, 0, 0, 0))

def as_matrix(quat):
    # expect w, x, y, z
    assert quat.shape == (4, 1)
    return (
        cs.MX.eye(3)
        + 2 * quat[0] * cs.skew(quat[1:4])
        + 2 * cs.mpower(cs.skew(quat[1:4]), 2)
    )

def export_model(g_val=9.81, return_params=False, cross_model=True, use_hybrid_mode=True):
    model_name = "quadrotor"
    mass_payload = 0.103 # 0.150
    cable_l = 0.561
    arm_length = 0.17
    prop_radius = 0.099
    inertia = np.zeros((3, 3))

    # voxl2
    ixx = 0.002404
    iyy = 0.00238
    izz = 0.0028
    inertia[0, 0] = ixx
    inertia[1, 1] = iyy
    inertia[2, 2] = izz
    mass_quad = 0.715
    kf =  0.88e-08
    km = 1.34e-10
    km_kf = km/kf 
        
    # TODO these cause the solver to fail
    # hover is 4126
    # kf = 4.37900e-9
    # km = 3.97e-11 # 0.07 * (2.0 * prop_radius) * kf
    # km = 0.07 * (2.0 * prop_radius) * kf

    # default
    # inertia[0, 0] = 2.64e-3
    # inertia[1, 1] = 2.64e-3
    # inertia[2, 2] = 4.96e-3
    # mass_quad = 0.5

    # kf = 5.55e-8
    # km = 0.07 * (2.0 * prop_radius) * kf

    invI = np.linalg.inv(inertia)

    hover_estimate = np.sqrt(((mass_payload + mass_quad)*g_val)/(kf*4.0))
    print(f"hover rpm estimate {hover_estimate}")
    print(f"hover force estimate {((mass_payload + mass_quad)*g_val)/4.0}")

    # set up states
    xL = MX.sym('xL')
    yL = MX.sym('yL')
    zL = MX.sym('zL')
    xLdot = MX.sym('xLdot')
    yLdot = MX.sym('yLdot')
    zLdot = MX.sym('zLdot')
    x = MX.sym('x')
    y = MX.sym('y')
    z = MX.sym('z')
    xdot = MX.sym('xdot')
    ydot = MX.sym('ydot')
    zdot = MX.sym('zdot')
    qw = MX.sym('qw')
    qx = MX.sym('qx')
    qy = MX.sym('qy')
    qz = MX.sym('qz')
    omega_x = MX.sym('omega_x')
    omega_y = MX.sym('omega_y')
    omega_z = MX.sym('omega_z')

    X = cs.vertcat(
        xL, yL, zL,                 # 0:3
        xLdot, yLdot, zLdot,        # 3:6
        x, y, z,                    # 6:9
        xdot, ydot, zdot,           # 9:12
        qw, qx, qy, qz,             # 12:16
        omega_x, omega_y, omega_z,  # 16:19
    )
    nx = X.shape[0]

    # parameters
    # states + inputs + if_taut
    ext_param = MX.sym('references', nx + 4 + 1, 1)

    # set up controls
    # motor rpms
    u1 = MX.sym('u1')
    u2 = MX.sym('u2')
    u3 = MX.sym('u3')
    u4 = MX.sym('u4')
    u = vertcat(u1, u2, u3, u4)

    if cross_model:
        dx = 0.21
        dy = 0.16
        ratio_x = 0.50
        ratio_y = 0.49
        dx_left = dx * ratio_x
        dx_right = dx * (1.0 - ratio_x)
        dy_front = dy * ratio_y
        dy_back = dy * (1.0 - ratio_y)

        
        moment_x = (dx_left * u1 + dx_left * u2
                    - dx_right * u3 - dx_right * u4)

        moment_y = (- dy_front * u1 + dy_back * u2
                    + dy_back * u3 - dy_front * u4)

        moment_z = km_kf * (u1 - u2 + u3 - u4)
        print("This")

    else:
        moment_x = kf * (cs.power(100.0 * u3, 2) - cs.power(100.0 * u4, 2)) * arm_length
        moment_y = kf * (cs.power(100.0 * u2, 2) - cs.power(100.0 * u1, 2)) * arm_length
        moment_z = km * (cs.power(100.0 * u1, 2) + cs.power(100.0 * u2, 2) - cs.power(100.0 * u3, 2) - cs.power(100.0 * u4, 2))

    F = (u1 + u2 + u3 + u4)
    # F = cs.exp(cs.log(kf)  + cs.log(cs.sumsqr(100.0 * u)))
    M = cs.vertcat(moment_x, moment_y, moment_z)

    norm_xi = cs.norm_2(cs.vertcat(xL - x, yL - y, zL - z))
    # xi = if_else(norm_xi < 0.002, cs.vertcat(0, 0, 0), cs.vertcat((xL - x)/norm_xi, (yL - y)/norm_xi, (zL - z)/norm_xi))
    xi = cs.vertcat((xL - x)/norm_xi, (yL - y)/norm_xi, (zL - z)/norm_xi)
    xidot = cs.vertcat((xLdot - xdot)/cable_l, (yLdot - ydot)/cable_l, (zLdot - zdot)/cable_l)
    xi_omega = cs.cross(xi, xidot)

    # Evaluate forces
    e3=np.array([[0.0],[0.0],[1.0]])
    g = g_val * e3

    quat = cs.vertcat(qw, qx, qy, qz)
    quad_force_vector = F * as_matrix(quat) @ e3
    quad_centrifugal_f = mass_quad * cable_l * (cs.dot(xi_omega, xi_omega))

    if use_hybrid_mode:
        temp_tension_vector = (mass_payload/(mass_payload + mass_quad)) * (-cs.dot(xi, quad_force_vector) + quad_centrifugal_f) * xi
        # tension_vector = if_else(norm_xi < (cable_l - 0.002), cs.vertcat(0, 0, 0), temp_tension_vector)
        tension_vector =  ext_param[-1]*temp_tension_vector
        print("True")

    else:
        print("False")
        tension_vector = (mass_payload/(mass_payload + mass_quad)) * (-cs.dot(xi, quad_force_vector) + quad_centrifugal_f) * xi

    accL = -tension_vector / mass_payload - g
    accQ = (quad_force_vector + tension_vector) / mass_quad - g


    qdot_helper = cs.vertcat(
                    cs.horzcat(0, -omega_x, -omega_y, -omega_z),
                    cs.horzcat(omega_x, 0, omega_z, -omega_y),
                    cs.horzcat(omega_y, -omega_z, 0, omega_x),
                    cs.horzcat(omega_z, omega_y, -omega_x, 0)
                    )
    K_quat = 2.0
    quaterr = 1.0 - cs.norm_2(quat)
    qdot = 0.5 * qdot_helper @ quat + K_quat * quaterr * quat

    omega = cs.vertcat(omega_x, omega_y, omega_z)
    pqrdot = invI @ (M - cs.cross(omega, inertia @ omega))

    f_expl_rpm = vertcat(
        xLdot,
        yLdot,
        zLdot,
        accL[0],
        accL[1],
        accL[2],
        xdot,
        ydot,
        zdot,
        accQ[0],
        accQ[1],
        accQ[2],
        qdot[0],
        qdot[1],
        qdot[2],
        qdot[3],
        pqrdot[0],
        pqrdot[1],
        pqrdot[2],
    )


    f_expl = f_expl_rpm
    Xdot = MX.sym('Xdot', nx, 1)

    # Formulation of the system for CBF x_dot = f(x) + g(x)u
    f_x = Function('f_x', [X, u, ext_param], [f_expl])
    g_x = Function('g_x', [X, ext_param], [jacobian(f_expl, u)])

    ## Projection of the payload respect to the camera and time derivative
    q_wb = quat
    w_t_wb = X[6:9, 0]

    # Internal states of the payload
    w_t = X[0:3, 0]

    # Camera position respect to the body frame
    camera_ori_w = 0.0
    camera_ori_x = -0.7071
    camera_ori_y = 0.7071
    camera_ori_z = 0.0
    camera_position_x = 0.0
    camera_position_y = 0.0
    camera_position_z = -0.075

    q_bc = vertcat(camera_ori_w, camera_ori_x, camera_ori_y, camera_ori_z)
    b_t_bc = vertcat(camera_position_x, camera_position_y, camera_position_z)
    q_wc = quat_multi(q_wb, q_bc)

    first_part = inverse_rot(q_wc, w_t - w_t_wb)
    second_part = inverse_rot(q_bc, b_t_bc)
    
    # Payload Projected to the camer frame
    projection_value = first_part - second_part

    # CBF and distance
    r_max = 0.30
    r_max_square = r_max*r_max

    # Distance center of the image
    distance = projection_value[0:2].T@projection_value[0:2]

    # CBF
    h = -distance + r_max_square

    ### First Derivative Jacobians
    dh_dx = jacobian(h, X)
    dot_distance = jacobian(distance, X)

    # Funtions for the distance to the center of the image  and the safety set
    h_f = Function('h_f', [X], [h])
    distance_f = Function('distance_f', [X], [distance])

    #  Funbtions of the Jacobians of the funtions mentioned before
    dh_dx_f = Function('dh_dx', [X], [dh_dx])
    ddistance_dx_f = Function('d_dot_f', [X], [dot_distance])

    p_ext = ext_param
    zz = []

    ## Section for lyapunov functions
    print("--------------------------")
    print(mass_quad)
    J = np.array([[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]])
    eigenvalues = np.linalg.eigvals(J)
    min_eigenvalue = np.min(eigenvalues)

    # Compute Constant values for the trasnlational controller
    c1 = 1
    kv_min = c1 + 1/4 + 0.1
    kp_min = (c1*(kv_min*kv_min) + 2*kv_min*c1 - c1*c1)/(mass_quad*(4*(kv_min - c1)-1))
    kp_min = 20.0
    print(kp_min)
    print(kv_min)
    print(c1)
    print("--------------------------")

    ## Compute minimiun values for the angular controller
    c2 = 0.05
    kw_min = (1/2)*c2 + (1/4) + 0.1
    kr_min = c2*(kw_min*kw_min)/(min_eigenvalue*(4*(kw_min - (1/2)*c2) - 1))
    kr_min = 20.0
    kr_min_real = 10.0
    print(kr_min)
    print(kw_min)
    print(c2)
    print("--------------------------")

    ## Compute minimum values for the controller associated to the payload trasnlational dynamics
    c3 = 2
    kv_min_payload = c3 + 1/4 + 0.1
    kp_min_payload = (c3*(kv_min_payload*kv_min_payload) + 2*kv_min_payload*c3 - c3*c3)/(mass_payload*(4*(kv_min_payload - c3)-1))
    #kp_min = 20
    print(kp_min_payload)
    print(kv_min_payload)
    print(c3)
    print("--------------------------")

    ## Payload states
    payload_position = vertcat(X[0], X[1], X[2])
    payload_position_references = vertcat(p_ext[0:3])

    payload_velocity =  vertcat(X[3], X[4], X[5])
    payload_velocity_references = vertcat(p_ext[3:6])

    # Quadrotor states
    quad_pos_states = vertcat(X[6], X[7], X[8])
    quad_pos_state_references = vertcat(p_ext[6:9])

    quad_vel_states = vertcat(X[9], X[10], X[11])
    quad_vel_state_references = vertcat(p_ext[9:12])

    quad_angular_states = vertcat(X[16], X[17], X[18])
    quad_angular_state_references = vertcat(p_ext[16:19])

    quat = vertcat(X[12], X[13], X[14], X[15])
    quat_references = vertcat(p_ext[12:16])

    # Computing translational error of the sytem quadrotor
    error_position_quad = quad_pos_states - quad_pos_state_references
    error_velocity_quad = quad_vel_states - quad_vel_state_references

    ## Creating lyapunov functions for trasnlation dynamics
    lyapunov_position_quad = (1/2)*kp_min*error_position_quad.T@error_position_quad + (1/2)*mass_quad*error_velocity_quad.T@error_velocity_quad
    lyapunov_position_quad_f = Function('lyapunov_position_quad_f', [X, p_ext], [lyapunov_position_quad]).expand()

    # Computing angular error of the system quadrotor
    angular_displacement_error = rotation_error_norm_f(quat_references, quat)
    angular_velocity_error = quad_angular_states - rotation_error_f(quat_references, quat)@quad_angular_state_references

    # Lyapunov Function angular error
    lyapunov_orientation_quad = kr_min*angular_displacement_error.T@angular_displacement_error + (1/2)*angular_velocity_error.T@J@angular_velocity_error
    lyapunov_orientation_quad_f = Function('lyapunov_orientation_quad_f', [X, p_ext], [lyapunov_orientation_quad]).expand()

    # Computing translational error of the sytem payload
    error_position_payload = payload_position - payload_position_references
    error_velocity_payload = payload_velocity - payload_velocity_references

    lyapunov_position_payload = (1/2)*kp_min_payload*error_position_payload.T@error_position_payload + (1/2)*mass_payload*error_velocity_payload.T@error_velocity_payload 
    lyapunov_position_payload_f = Function('lyapunov_position_payload_f', [X, p_ext], [lyapunov_position_payload]).expand()

    # Complete lyapunov function
    lyapunov = lyapunov_position_quad + lyapunov_position_payload + lyapunov_orientation_quad
    lyapunov_f = Function('lyapunov_f', [X, p_ext], [lyapunov]).expand()

    model = AcadosModel()
    model.f_impl_expr = Xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = X
    model.u = u
    model.xdot = Xdot
    model.z = zz
    model.p = p_ext
    model.name = model_name

    return model, f_x, g_x, h_f, distance_f, dh_dx_f, ddistance_dx_f, lyapunov_f