# HW4 Bundle Adjustment

*Yucong Chen 2019533079*

**Files:**

> main.py
>
> vis.py
>
> bundle_adj.py
>
> add_noise.py
>

**Usages:**

> firstly run:
>
> python3 main.py
>
> then:
>
> python3 add_noise.py
>
> finally:
>
> python3 bundle_adj.py
>
> Visualization on point cloud can be found in vis.py



## Task 1: Projection and Back-projection

#### projection:

$$\left(\begin{matrix}u\\ v\\ 1\end{matrix}\right) = \left(\begin{matrix}\alpha&\gamma&u_0&0\\0&\beta&u_0&0\\ 0&0&1&0\end{matrix}\right)\left(\begin{matrix}R&t\\0 &1\end{matrix}\right)\left(\begin{matrix}X\\ Y\\ Z\\1\end{matrix}\right)$$

#### back-projection:

$$ \left(\begin{matrix}X\\ Y\\ Z\\1\end{matrix}\right)= \left(\begin{matrix}R&t\\0 &1\end{matrix}\right)^{-1}\left(\begin{matrix}\alpha&\gamma&u_0&0\\0&\beta&u_0&0\\ 0&0&1&0\end{matrix}\right)^{-1}\left(\begin{matrix}u\\ v\\ 1\end{matrix}\right)$$

#### Results:

###### World points:<img src="/Users/chenyc/Documents/CS284/hw4/gt_world_points.png" alt="gt_world_points" style="zoom:30%;" />

###### C1 view -- C9 view (C5 omitted):
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c1.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c2.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c3.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c4.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c6.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c7.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c8.png" alt="gt_image_coords_c1" style="zoom:30%;" />
<img src="/Users/chenyc/Documents/CS284/hw4/gt_image_coords_c9.png" alt="gt_image_coords_c1" style="zoom:30%;" />

### Task 2: Add Noise

Add noise to world points and poses as follows (red is groundtruth and blue is perturbed data):
<img src="/Users/chenyc/Documents/CS284/hw4/noise_world_points.png" alt="noise_world_points" style="zoom:25%;" />

<img src="/Users/chenyc/Documents/CS284/hw4/wh1.png" alt="wh2" style="zoom:30%;" />

********

<img src="/Users/chenyc/Documents/CS284/hw4/wh2.png" alt="wh2" style="zoom:30%;" />

And also add noise to the pixel measurement (which does not be visualized here)



### Task 3: Bundle Adjustment

#### Calculate Jacobians Numerically:

```python
def Jacobian(input, f0):
    eps = 1e-10
    J = np.zeros((f0.shape[0], input.shape[0]))
    for i in range(input.shape[0]):
        increased_input = input.copy()
        increased_input[i] += eps
        f1 = f(increased_input)
        J[:, i] = (f1 - f0) / eps
    return J
```

#### Apply Gaussian-Newton Optimization:

```python
def GN(X, input):
    for i in tqdm(range(10)):
        f0 = f(input)
        e0 = X - f0
        J = Jacobian(input, f0)
        H = J.T @ J
        delta = np.linalg.inv(H + np.eye(H.shape[0])) @ J.T @ e0
        input += 0.5*delta
    return input
```

Notice that I added identity matrix to $H$ in the updating calculation to increase the stability and convergency. And also nomalized coordinates to eliminate $K$.

#### Results:

<img src="/Users/chenyc/Documents/CS284/hw4/loss.png" alt="loss" style="zoom:24%;" />

The curve shows the $\|e_0\|_2$, and its convergency after just few iterations.

The follows table shows the world points after doing bundle adjustment:

<img src="/Users/chenyc/Documents/CS284/hw4/bundle_adj_world_points.png" alt="bundle_adj_world_points" style="zoom:24%;" />

which is clearly more close to the groundtruth comparing with noisy data. And Here is the visualization:

<img src="/Users/chenyc/Documents/CS284/hw4/adj_p.png" alt="adj_p" style="zoom:50%;" />

where red is gt, blue is noisy data and green is bundle adjustment result.



### Discussion:

- The original Gaussian-Newton method is unstable, and can be improved by adding an identity matrix to the $J^TJ$;
- The epsilon for numerical method is somehow important to maintain the stability;
- The learning rate is very important for both getting a good result and converging fast.

