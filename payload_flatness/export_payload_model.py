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
    inertia[0, 0] = 0.002404
    inertia[1, 1] = 0.00238
    inertia[2, 2] = 0.0028
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

    model = AcadosModel()
    model.f_impl_expr = Xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = X
    model.u = u
    model.xdot = Xdot
    model.z = zz
    model.p = p_ext
    model.name = model_name

    return model, f_x, g_x, h_f, distance_f, dh_dx_f, ddistance_dx_f