/*
 * Phase 2B-3: PyTorch/Python Bindings
 * ====================================
 * 
 * Exposes CUDA kernels to Python through pybind11.
 * Uses torch::extension to seamlessly handle tensor conversion.
 */

#include <torch/extension.h>

// Forward declarations of kernel launch functions
torch::Tensor launch_advect_2d(torch::Tensor density, torch::Tensor velocity, float dt);
torch::Tensor launch_advect_velocity_2d(torch::Tensor velocity, float dt);
torch::Tensor launch_advect_3d(torch::Tensor density, torch::Tensor velocity, float dt);


// ═══════════════════════════════════════════════════════════════════════════
// Python Module Definition
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        TensorNet CUDA Extension
        ========================
        
        High-performance CUDA kernels for tensor network CFD operations.
        
        Functions:
            advect_2d: Semi-Lagrangian advection for 2D scalar fields
            advect_velocity_2d: Self-advection for 2D velocity fields
            advect_3d: Semi-Lagrangian advection for 3D scalar fields
    )pbdoc";
    
    m.def(
        "advect_2d",
        &launch_advect_2d,
        R"pbdoc(
            Advect a 2D scalar field by a velocity field.
            
            Uses Semi-Lagrangian method with bilinear interpolation.
            
            Args:
                density (Tensor): Input scalar field [H, W], must be CUDA float32
                velocity (Tensor): Velocity field [2, H, W], must be CUDA float32
                dt (float): Time step
                
            Returns:
                Tensor: Advected density field [H, W]
                
            Example:
                >>> import tensornet_cuda
                >>> density = torch.rand(512, 512, device='cuda', dtype=torch.float32)
                >>> velocity = torch.rand(2, 512, 512, device='cuda', dtype=torch.float32)
                >>> result = tensornet_cuda.advect_2d(density, velocity, 0.01)
        )pbdoc",
        py::arg("density"),
        py::arg("velocity"),
        py::arg("dt")
    );
    
    m.def(
        "advect_velocity_2d",
        &launch_advect_velocity_2d,
        R"pbdoc(
            Advect a 2D velocity field by itself (self-advection).
            
            Used for: u^{n+1} = u^n - dt * (u · ∇)u
            
            Args:
                velocity (Tensor): Velocity field [2, H, W], must be CUDA float32
                dt (float): Time step
                
            Returns:
                Tensor: Advected velocity field [2, H, W]
        )pbdoc",
        py::arg("velocity"),
        py::arg("dt")
    );
    
    m.def(
        "advect_3d",
        &launch_advect_3d,
        R"pbdoc(
            Advect a 3D scalar field by a velocity field.
            
            Uses Semi-Lagrangian method with trilinear interpolation.
            
            Args:
                density (Tensor): Input scalar field [D, H, W], must be CUDA float32
                velocity (Tensor): Velocity field [3, D, H, W], must be CUDA float32
                dt (float): Time step
                
            Returns:
                Tensor: Advected density field [D, H, W]
        )pbdoc",
        py::arg("density"),
        py::arg("velocity"),
        py::arg("dt")
    );
}
