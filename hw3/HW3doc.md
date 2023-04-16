# HW3 Camera Calibration 

*Yucong Chen 2019533079*

## Part A: Zhengyou Zhang's Method

#### Algorithm

Firstly find the corner points in images by OpenCV functions. I further added the subpixel optimization process to get precise corner locations.

<img src="/Users/chenyc/Documents/CS284/hw3/corner_detection.png" alt="corner_detection" style="zoom:20%;" />

Now we have the equation between world points and pixel coords:

$$\left(\begin{matrix}u\\ v\\ 1\end{matrix}\right) = \left(\begin{matrix}\alpha&\gamma&u_0&0\\0&\beta&u_0&0\\ 0&0&1&0\end{matrix}\right)\left(\begin{matrix}R&t\\0 &1\end{matrix}\right)\left(\begin{matrix}X\\ Y\\ Z\\1\end{matrix}\right)$$

The objection is to find the intrinsics matrix $K$ and extrinsics $R$ and $t$.

From this, first of all I find the Homography matrix $H$ by solving the linear optimization problem such that:

$$\left(\begin{matrix}u\\ v\\ 1\end{matrix}\right) = H\left(\begin{matrix}X\\ Y\\ Z\\1\end{matrix}\right)$$

Then let $B = K^{-T}K^{-1}$, and use $H$ to construct vectors $v_{11}, v_{12}, v_{22}$. After that we build a equation set:

$$\left(\begin{matrix}v_{12}^T\\(v_{11}-v_{12})^T\end{matrix}\right)b = 0$$

Where $b$ is flattened $B$. Then use SVD to solve it (constraint $b$'s norm to be $1$). So that we can have $B$.

Now we can get $K$ from $B$. After that, from the first equation we can also have the corresponding $R$ and $t$.

#### Results

Here is the result of the two cameras:

$$K_A = \left(\begin{matrix}1470.4&1.7&1229.1\\
       			0&1469.9&1033.7\\
       			0 &0 &1\end{matrix}\right)$$



$$K_B = \left(\begin{matrix}1461.7&3.3&1219.7\\
       			0&1459.7&1022.3\\
       			0 &0 &1\end{matrix}\right)$$

Reprojection Error ($l2$) of $A$: 0.106

Reprojection Error ($l2$) of $B$: 0.105

And the extrinsics are saved in the `outputs/partA`



## Part B: Hand-Eye Calibration

#### Algorithm (Bonus*)

Similar to PartA to find the corners:

<img src="/Users/chenyc/Documents/CS284/hw3/corner_detection_b.png" alt="corner_detection_b" style="zoom:20%;" />

From the Problem settings we get this formulation (All are $SE(3)$):

$$A_iA_j^{-1}Y = YB_iB_j^{-1}$$

And now the formula can be written as:

$$\left(\begin{matrix}R_A&t_A\\0&1\end{matrix}\right)\left(\begin{matrix}R_X&t_X\\0&1\end{matrix}\right)=\left(\begin{matrix}R_X&t_X\\0&1\end{matrix}\right)\left(\begin{matrix}R_B&t_B\\0&1\end{matrix}\right)$$

Expand and get:

$$\begin{cases}R_AR_X = R_XR_B\\\ \\R_At_X + t_A= R_Xt_B+t_X\end{cases}$$

Define the Kronecker Product operation as $\otimes$

From the first equation, we can get:

$$(R_A\otimes I - I\otimes R_B^T)\,vec(R_X) = 0$$

Solve it using SVD, then we get $R_X$ and then we put it into the second equation, we now have $t_X$ (least square)

$$Y = \left(\begin{matrix}R_X&t_X\\0&1\end{matrix}\right)$$

#### Results

From Camera A (left) and Camera B (right), we can get $Y$ as follows:

<img src="/Users/chenyc/Documents/CS284/hw3/outputs/partB/Y.png" alt="Y" style="zoom:30%;" />

Reprojection Error ($l2$) of $A$: 0.068

Reprojection Error ($l2$) of $B$: 0.075

As we can see, the two matrices are similar to each other.

The difference is $\|Y_A^{-1}Y_A - I\|_2 = 0.0253$

And as the demonstration here:



<img src="/Users/chenyc/Documents/CS284/hw3/outputs/partB/frames.png" alt="frames" style="zoom:20%;" />

Where read cameras are from A and blue cameras are from B. The right most coordinate frame is checkerboard frame while those two closely aligned frames are marker frames from $Y_A$ and $Y_B$. We can see that they are very close to each other.
