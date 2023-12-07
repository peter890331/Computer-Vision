###
### This homework is modified from CS231.
###

import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    U, sigma, V_transpose = np.linalg.svd(E)            # 將 essential matrix 做 SVD 分解為 U, sigma, V_transpose

    #Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])    # 定義 W 矩陣

    Q1 = U.dot(W.dot(V_transpose))                      # Q1 = U W VT
    Q2 = U.dot(W.T.dot(V_transpose))                    # Q2 = U WT VT

    T1 = U[:, 2]                                        # T = u3
    T2 = -U[:, 2]                                       # T = -u3

    R1 = (np.linalg.det(Q1) * Q1).T                     # R1 = det(Q1) · Q1
    R2 = (np.linalg.det(Q2) * Q2).T                     # R2 = det(Q2) · Q2

    RT = np.array([                                     # vstack rotation (R) and translation (T)
        np.vstack([R1, T1]).T,
        np.vstack([R1, T2]).T,
        np.vstack([R2, T1]).T,
        np.vstack([R2, T2]).T
    ])

    return RT                                           # RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    pi = image_points.copy()                            # pi = image_points
    Mi = camera_matrices.copy()                         # Mi = camera_matrices

    A1 = (pi[:, 1] * Mi[:, 2, :].T).T - Mi[:, 1, :]     # v_n M_n^3 - M_n^2
    A2 = Mi[:, 0, :] - (pi[:, 0] * Mi[:, 2, :].T).T     # M_n^1 - u_n M_n^3
    A = np.vstack([A1, A2])

    U, sigma, V_transpose = np.linalg.svd(A)            # solve P by SVD
    point_3d = V_transpose[3, :].copy()
    point_3d /= point_3d[-1]
    return point_3d[:-1]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''

def reprojection_error(point_3d, image_points, camera_matrices):
    pi = image_points.copy()                            # pi = image_points
    Mi = camera_matrices.copy()                         # Mi = camera_matrices
    P = np.hstack([point_3d.copy(), 1])                 # P = [point_3d\\ 1]

    y = np.matmul(Mi, P)                                # y = M_i P
    y = y.T
    pi_prime = y / y[-1, :]                             # pi_prime = 1/y_3 [y_1\\ y_2]

    ei = (pi_prime[:-1, :].T - pi).reshape(2 * pi.shape[0], )       # ei = pi_prime - pi
    return ei

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''

def jacobian(point_3d, camera_matrices):
    P = np.hstack([point_3d.copy(), 1])                 # P = [point_3d\\ 1]
    Mi = camera_matrices.copy()                         # Mi = camera_matrices

    numerator = (np.matmul(Mi[:, 2, :], P)) ** 2        # jacobian 所有偏微分的共同分母

    Jx1 = Mi[:, 0, 0] * np.matmul(Mi[:, 2, [1, 2, 3]], P[[1, 2, 3]]) - Mi[:, 2, 0] * np.matmul(Mi[:, 0, [1, 2, 3]], P[[1, 2, 3]])       # jacobian 所有偏微分中每一個 partial X 元素的第一項
    Jx2 = Mi[:, 0, 1] * np.matmul(Mi[:, 2, [0, 2, 3]], P[[0, 2, 3]]) - Mi[:, 2, 1] * np.matmul(Mi[:, 0, [0, 2, 3]], P[[0, 2, 3]])       # jacobian 所有偏微分中每一個 partial Y 元素的第一項
    Jx3 = Mi[:, 0, 2] * np.matmul(Mi[:, 2, [0, 1, 3]], P[[0, 1, 3]]) - Mi[:, 2, 2] * np.matmul(Mi[:, 0, [0, 1, 3]], P[[0, 1, 3]])       # jacobian 所有偏微分中每一個 partial Z 元素的第一項

    Jy1 = Mi[:, 1, 0] * np.matmul(Mi[:, 2, [1, 2, 3]], P[[1, 2, 3]]) - Mi[:, 2, 0] * np.matmul(Mi[:, 1, [1, 2, 3]], P[[1, 2, 3]])       # jacobian 所有偏微分中每一個 partial X 元素的第二項
    Jy2 = Mi[:, 1, 1] * np.matmul(Mi[:, 2, [0, 2, 3]], P[[0, 2, 3]]) - Mi[:, 2, 1] * np.matmul(Mi[:, 1, [0, 2, 3]], P[[0, 2, 3]])       # jacobian 所有偏微分中每一個 partial Y 元素的第二項
    Jy3 = Mi[:, 1, 2] * np.matmul(Mi[:, 2, [0, 1, 3]], P[[0, 1, 3]]) - Mi[:, 2, 2] * np.matmul(Mi[:, 1, [0, 1, 3]], P[[0, 1, 3]])       # jacobian 所有偏微分中每一個 partial Z 元素的第二項

    Jx = np.vstack([[Jx1], [Jx2], [Jx3]])               # 組合 jacobian 所有偏微分的第一項
    Jy = np.vstack([[Jy1], [Jy2], [Jy3]])               # 組合 jacobian 所有偏微分的第二項
    Jx = np.divide(Jx, numerator).T                     # 同除共同分母
    Jy = np.divide(Jy, numerator).T                     # 同除共同分母

    jacobian = np.zeros((2 * Jx.shape[0], Jy.shape[1])) # 組合 jacobian 矩陣
    jacobian[0::2, :] = Jx
    jacobian[1::2, :] = Jy

    return jacobian

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    pi = image_points.copy()                            # pi = image_points
    Mi = camera_matrices.copy()                         # Mi = camera_matrices

    iterations = 10                                     # run the optimization for 10 iterations

    estimated_3d_point = linear_estimate_3d_point(pi, Mi)                                                                       # linear_estimate_3d_point()

    for i in range(iterations):
        J = jacobian(estimated_3d_point, Mi)                                                                                    # jacobian()
        reprojection_error_ = reprojection_error(estimated_3d_point, pi, Mi)                                                    # reprojection error()
        estimated_3d_point = estimated_3d_point - np.matmul(np.matmul(np.linalg.inv(J.T.dot(J)), J.T), reprojection_error_)     # P = P - (J^T J)^-1 J^T e

    return estimated_3d_point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    estimate_initial_RT_ = estimate_initial_RT(E)       # 呼叫四種可能的 RT

    correct_RT_temp = [0, 0, 0, 0]                      # 建立投票矩陣

    M1 = K.dot(np.append(np.eye(3), np.zeros((3, 1)), axis=1))      # 第一個相機的相機矩陣

    for i in range(image_points.shape[0]):
        for j in range(estimate_initial_RT_.shape[0]):
            M2 = K.dot(estimate_initial_RT_[j])                     # 透過四種 RT 轉換的第二個相機的相機矩陣
            M = np.array((M1, M2))                                  # 合併兩種相機矩陣

            X = linear_estimate_3d_point(image_points[i], M)        # 估計第一個3D點
            X2 = estimate_initial_RT_[j].dot(np.append(X, 1).T)     # 透過四種 RT 轉換的第二個3D點

            if X2[2] > 0 and X[2] > 0:                              # 檢查兩個3D點的Z軸是否都為正
                correct_RT_temp[j] += 1                             # 在四種 RT 的投票矩陣投票

    RT = estimate_initial_RT_[np.argmax(correct_RT_temp)]           # 得到真正的RT

    return RT


if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir, 'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir, 'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
                           [0.1019, 0.9948, 0.0045, -0.0089],
                           [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(), camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[154.33943931, 0., -22.42541691],
                                  [0., 154.33943931, 36.51165089],
                                  [141.87950588, -14.27738422, -56.20341644],
                                  [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(estimated_3d_point_linear, unit_test_image_matches, unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(estimated_3d_point_nonlinear, unit_test_image_matches, unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E, np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length, fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
