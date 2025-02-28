import matplotlib.pyplot as plt
import numpy as np


def get_joint(num_samples, do_B=None):
    # A
    A = np.random.normal(size=(num_samples, 1))

    # B
    if do_B is None:
        B = A + 0.5 * np.random.normal(size=(num_samples, 1))
    else:
        B = do_B * np.ones((num_samples, 1))

    # C
    C = B + 0.5 * np.random.normal(size=(num_samples, 1))

    return A, B, C


def get_joint_cond_B(num_samples, margins_of_B, do_B=None):
    # A
    A = np.random.normal(size=(num_samples, 1))

    # B
    if do_B is None:
        B = A + 0.1 * np.random.normal(size=(num_samples, 1))
    else:
        B = do_B * np.ones((num_samples, 1))

    # C
    C = B + 0.1 * np.random.normal(size=(num_samples, 1))

    # condition on B (get only the index of B where the value is inside the margins)
    idx = np.where((B >= margins_of_B[0]) & (B <= margins_of_B[1]))[0]

    return A[idx], B[idx], C[idx]


def main():
    # define experiment parameters
    NUM_SAMPLES = 1000
    DO_B = 3

    # no intervention
    A, B, C = get_joint(NUM_SAMPLES)
    plt.hist(A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("No Intervention")
    plt.legend()
    plt.show()

    # do(B) intervention
    A, B, C = get_joint(NUM_SAMPLES, do_B=DO_B)
    plt.hist(A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Do(B={})".format(DO_B))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
