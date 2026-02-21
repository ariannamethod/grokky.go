package main

// accel.go — BLAS acceleration stubs for Linux (pure Go fallback)
// On macOS, this would use Apple Accelerate via CGO.
// On Linux without OpenBLAS, we fall back to pure Go matmul.

func AccelMatMulQ4_0(out []float32, w []byte, x []float32, rows, cols int) {
	MatMulQ4_0(out, w, x, rows, cols)
}

func AccelMatMulQ8_0(out []float32, w []byte, x []float32, rows, cols int) {
	MatMulQ8_0(out, w, x, rows, cols)
}

func AccelMatMulQ4K(out []float32, w []byte, x []float32, rows, cols int) {
	MatMulQ4_K(out, w, x, rows, cols)
}

func AccelMatMulQ6K(out []float32, w []byte, x []float32, rows, cols int) {
	MatMulQ6_K(out, w, x, rows, cols)
}
