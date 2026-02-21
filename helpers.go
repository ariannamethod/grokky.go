package main

// helpers.go — shared utility functions for GGUF tensor loading and RoPE
// Extracted from deepseek.go model.go to avoid LlamaModel conflicts.

import (
	"fmt"
	"math"
)

// applyRoPE applies rotary position encoding (Grok-compatible signature)
func applyRoPE(vec []float32, pos int, cosCache, sinCache []float32, headDim int) {
	half := headDim / 2
	cacheOff := pos * half
	for i := 0; i < half; i++ {
		x0 := vec[i]
		x1 := vec[i+half]
		c := cosCache[cacheOff+i]
		si := sinCache[cacheOff+i]
		vec[i] = x0*c - x1*si
		vec[i+half] = x0*si + x1*c
	}
}

// getF32Tensor loads a tensor and dequantizes to float32
func getF32Tensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, err
	}
	switch info.Type {
	case ggmlTypeF32:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			out[i] = math.Float32frombits(
				uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
					uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24)
		}
		return out, nil
	case ggmlTypeF16:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			h := uint16(data[i*2]) | uint16(data[i*2+1])<<8
			out[i] = half2float(h)
		}
		return out, nil
	case ggmlTypeQ4_0:
		return DequantQ4_0(data, expectedSize), nil
	case ggmlTypeQ5_0:
		return DequantQ5_0(data, expectedSize), nil
	case ggmlTypeQ8_0:
		return DequantQ8_0(data, expectedSize), nil
	case ggmlTypeQ4_K:
		return DequantQ4_K(data, expectedSize), nil
	case ggmlTypeQ6_K:
		return DequantQ6_K(data, expectedSize), nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %d for %s", info.Type, name)
	}
}

// getF32TensorOptional loads a tensor if it exists, nil if not found
func getF32TensorOptional(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	_, _, err := gguf.GetTensor(name)
	if err != nil {
		return nil, nil
	}
	return getF32Tensor(gguf, name, expectedSize)
}

// getRawTensor returns raw bytes + type for a tensor
func getRawTensor(gguf *GGUFFile, name string) ([]byte, uint32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, 0, err
	}
	return data, info.Type, nil
}

// embedLookupInto extracts an embedding row into a pre-allocated buffer
func embedLookupInto(out []float32, data []byte, dtype uint32, token, dim int) {
	switch dtype {
	case ggmlTypeQ4_0:
		blocksPerRow := dim / q4BlockSize
		bytesPerRow := blocksPerRow * q4BytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4BytesPerBlock
			DequantQ4_0Block(data[blockOff:blockOff+q4BytesPerBlock], out[b*q4BlockSize:])
		}
	case ggmlTypeQ8_0:
		blocksPerRow := dim / q8BlockSize
		bytesPerRow := blocksPerRow * q8BytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q8BytesPerBlock
			DequantQ8_0Block(data[blockOff:blockOff+q8BytesPerBlock], out[b*q8BlockSize:])
		}
	case ggmlTypeF16:
		off := token * dim * 2
		for i := 0; i < dim; i++ {
			h := uint16(data[off+i*2]) | uint16(data[off+i*2+1])<<8
			out[i] = half2float(h)
		}
	case ggmlTypeF32:
		off := token * dim * 4
		for i := 0; i < dim; i++ {
			out[i] = math.Float32frombits(
				uint32(data[off+i*4]) | uint32(data[off+i*4+1])<<8 |
					uint32(data[off+i*4+2])<<16 | uint32(data[off+i*4+3])<<24)
		}
	default:
		for i := 0; i < dim; i++ {
			out[i] = 0
		}
	}
}

// matmulDispatch dispatches to the right matmul based on tensor type
func matmulDispatch(out []float32, w []byte, wtype uint32, x []float32, rows, cols int) {
	switch wtype {
	case ggmlTypeQ4_0:
		AccelMatMulQ4_0(out, w, x, rows, cols)
	case ggmlTypeQ5_0:
		MatMulQ5_0(out, w, x, rows, cols)
	case ggmlTypeQ8_0:
		AccelMatMulQ8_0(out, w, x, rows, cols)
	case ggmlTypeF16:
		MatMulF16(out, w, x, rows, cols)
	case ggmlTypeF32:
		f32 := make([]float32, len(w)/4)
		for i := range f32 {
			f32[i] = math.Float32frombits(
				uint32(w[i*4]) | uint32(w[i*4+1])<<8 |
					uint32(w[i*4+2])<<16 | uint32(w[i*4+3])<<24)
		}
		MatMulF32(out, f32, x, rows, cols)
	case ggmlTypeQ4_K:
		AccelMatMulQ4K(out, w, x, rows, cols)
	case ggmlTypeQ6_K:
		AccelMatMulQ6K(out, w, x, rows, cols)
	default:
		fmt.Printf("[grok] WARNING: unsupported matmul type %d\n", wtype, rows, cols)
	}
}
