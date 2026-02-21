package main

// model.go — Grok MoE forward pass for grokky.go
//
// Grok-1 architecture (adapted for small distills):
//   MoE: N experts, top-k routing, optional shared expert
//   GELU activation (not SiLU)
//   Double pre-norm: RMSNorm before AND after each sub-layer
//   Soft attention clamping: clamp * tanh(logits / clamp)
//   GQA (Grouped Query Attention) with RoPE
//
// This is not inference. This is rebellion.

import (
	"fmt"
	"math"
	"sort"
)

// GrokModel is a loaded Grok MoE model ready for inference
type GrokModel struct {
	Config  GrokConfig
	Weights GrokWeights
	State   GrokState
}

// GrokConfig holds model dimensions + MoE config
type GrokConfig struct {
	NumLayers  int
	EmbedDim   int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	VocabSize  int
	SeqLen     int
	IntermSize int
	RMSNormEps float32
	RopeTheta  float32

	// MoE config
	NumExperts       int
	NumExpertsPerTok int
	HasSharedExpert  bool

	// Grok-specific
	UseGELU       bool
	DoublePrenorm bool
	AttnClamp     float32 // 0 = off, 30 = Grok-1

	// nanollama flags
	QKNorm        bool
	RopeConjugate bool
}

// ExpertWeights holds weights for one expert FFN
type ExpertWeights struct {
	WGate     []byte
	WGateType uint32
	WUp       []byte
	WUpType   uint32
	WDown     []byte
	WDownType uint32
}

// GrokLayerWeights holds weights for one Grok transformer layer
type GrokLayerWeights struct {
	// Pre-norms
	AttnNorm []float32
	FFNNorm  []float32
	// Post-norms (double pre-norm, Grok-1 style)
	AttnPostNorm []float32 // nil if not double prenorm
	FFNPostNorm  []float32

	// Attention projections
	WQ     []byte
	WQType uint32
	WK     []byte
	WKType uint32
	WV     []byte
	WVType uint32
	WO     []byte
	WOType uint32

	// MoE router [num_experts, dim]
	WRouter     []byte
	WRouterType uint32

	// Experts
	Experts []ExpertWeights

	// Shared expert (always-on, optional)
	SharedExpert *ExpertWeights
}

// GrokWeights holds all weight tensors
type GrokWeights struct {
	TokenEmbed   []byte
	TokenEmbType uint32
	OutputNorm   []float32
	Output       []byte
	OutputType   uint32
	Layers       []GrokLayerWeights
}

// GrokState holds runtime buffers and KV cache
type GrokState struct {
	X      []float32 // current hidden state
	XB     []float32 // buffer after norm
	XB2    []float32 // second buffer (attention output)
	XB3    []float32 // third buffer (for post-norm)
	HB     []float32 // MLP hidden buffer
	HB2    []float32 // MLP gate buffer
	HBShared  []float32 // shared expert buffer
	HB2Shared []float32
	Q      []float32
	K      []float32
	V      []float32
	Att    []float32
	Logits []float32

	KeyCache   []float32
	ValueCache []float32
	CosCache   []float32
	SinCache   []float32
	EmbBuf     []float32

	// MoE buffers
	RouterLogits  []float32 // [num_experts]
	ExpertOutputs []float32 // [dim] accumulated expert output

	Pos int
}

// LoadGrokModel builds a GrokModel from a parsed GGUF file
func LoadGrokModel(gguf *GGUFFile) (*GrokModel, error) {
	meta := &gguf.Meta

	cfg := GrokConfig{
		NumLayers:  meta.NumLayers,
		EmbedDim:   meta.EmbedDim,
		NumHeads:   meta.NumHeads,
		NumKVHeads: meta.NumKVHeads,
		HeadDim:    meta.HeadDim,
		VocabSize:  meta.VocabSize,
		SeqLen:     meta.SeqLen,
		IntermSize: meta.IntermSize,
		RMSNormEps: meta.RMSNormEps,
		RopeTheta:  meta.RopeTheta,
		QKNorm:     meta.QKNorm,

		NumExperts:       meta.NumExperts,
		NumExpertsPerTok: meta.NumExpertsPerTok,
	}

	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}
	if cfg.SeqLen > 2048 {
		fmt.Printf("[grok] capping seq_len from %d to 2048\n", cfg.SeqLen)
		cfg.SeqLen = 2048
	}
	if cfg.NumExperts == 0 {
		cfg.NumExperts = 1 // dense fallback
		cfg.NumExpertsPerTok = 1
	}
	if cfg.NumExpertsPerTok == 0 {
		cfg.NumExpertsPerTok = 2
	}

	// Read Grok-specific metadata
	cfg.UseGELU = meta.UseGELU
	cfg.DoublePrenorm = meta.DoublePrenorm
	cfg.AttnClamp = meta.AttnClamp
	cfg.HasSharedExpert = meta.HasSharedExpert

	w, err := loadGrokWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	state := allocGrokState(&cfg)
	precomputeGrokRoPE(&state, &cfg)

	model := &GrokModel{Config: cfg, Weights: *w, State: state}

	fmt.Printf("[grok] loaded: %dL %dD %dH %dKV experts=%dx%d shared=%v gelu=%v double_prenorm=%v clamp=%.0f\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads,
		cfg.NumExperts, cfg.NumExpertsPerTok, cfg.HasSharedExpert,
		cfg.UseGELU, cfg.DoublePrenorm, cfg.AttnClamp)

	return model, nil
}

// loadGrokWeights maps GGUF tensors to GrokWeights
func loadGrokWeights(gguf *GGUFFile, cfg *GrokConfig) (*GrokWeights, error) {
	w := &GrokWeights{}
	var err error

	// Token embedding
	emb, embInfo, err := gguf.GetTensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	w.TokenEmbed = emb
	w.TokenEmbType = embInfo.Type

	// Output norm
	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm: %w", err)
	}

	// Output head
	outData, outInfo, err := gguf.GetTensor("output.weight")
	if err != nil {
		outData = w.TokenEmbed
		outInfo = embInfo
		fmt.Printf("[grok] output.weight not found, using tied embeddings\n")
	}
	w.Output = outData
	w.OutputType = outInfo.Type

	// Per-layer weights
	w.Layers = make([]GrokLayerWeights, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		pfx := fmt.Sprintf("blk.%d.", i)
		l := &w.Layers[i]

		// Norms
		l.AttnNorm, err = getF32Tensor(gguf, pfx+"attn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		l.FFNNorm, err = getF32Tensor(gguf, pfx+"ffn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}

		// Post-norms (optional, Grok double prenorm)
		if cfg.DoublePrenorm {
			l.AttnPostNorm, _ = getF32TensorOptional(gguf, pfx+"attn_post_norm.weight", cfg.EmbedDim)
			l.FFNPostNorm, _ = getF32TensorOptional(gguf, pfx+"ffn_post_norm.weight", cfg.EmbedDim)
		}

		// Attention
		l.WQ, l.WQType, err = getRawTensor(gguf, pfx+"attn_q.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		l.WK, l.WKType, err = getRawTensor(gguf, pfx+"attn_k.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		l.WV, l.WVType, err = getRawTensor(gguf, pfx+"attn_v.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		l.WO, l.WOType, err = getRawTensor(gguf, pfx+"attn_output.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		// MoE router
		if cfg.NumExperts > 1 {
			l.WRouter, l.WRouterType, err = getRawTensor(gguf, pfx+"ffn_gate_inp.weight")
			if err != nil {
				return nil, fmt.Errorf("layer %d ffn_gate_inp (router): %w", i, err)
			}
		}

		// Experts
		l.Experts = make([]ExpertWeights, cfg.NumExperts)
		for e := 0; e < cfg.NumExperts; e++ {
			var ePfx string
			if cfg.NumExperts > 1 {
				ePfx = fmt.Sprintf("%sffn_gate.%d.weight", pfx, e)
			} else {
				ePfx = pfx + "ffn_gate.weight"
			}
			l.Experts[e].WGate, l.Experts[e].WGateType, err = getRawTensor(gguf, ePfx)
			if err != nil {
				return nil, fmt.Errorf("layer %d expert %d gate: %w", i, e, err)
			}
			if cfg.NumExperts > 1 {
				ePfx = fmt.Sprintf("%sffn_up.%d.weight", pfx, e)
			} else {
				ePfx = pfx + "ffn_up.weight"
			}
			l.Experts[e].WUp, l.Experts[e].WUpType, err = getRawTensor(gguf, ePfx)
			if err != nil {
				return nil, fmt.Errorf("layer %d expert %d up: %w", i, e, err)
			}
			if cfg.NumExperts > 1 {
				ePfx = fmt.Sprintf("%sffn_down.%d.weight", pfx, e)
			} else {
				ePfx = pfx + "ffn_down.weight"
			}
			l.Experts[e].WDown, l.Experts[e].WDownType, err = getRawTensor(gguf, ePfx)
			if err != nil {
				return nil, fmt.Errorf("layer %d expert %d down: %w", i, e, err)
			}
		}

		// Shared expert (optional)
		if cfg.HasSharedExpert {
			se := &ExpertWeights{}
			se.WGate, se.WGateType, err = getRawTensor(gguf, pfx+"ffn_gate_shexp.weight")
			if err == nil {
				se.WUp, se.WUpType, _ = getRawTensor(gguf, pfx+"ffn_up_shexp.weight")
				se.WDown, se.WDownType, _ = getRawTensor(gguf, pfx+"ffn_down_shexp.weight")
				l.SharedExpert = se
			}
		}
	}

	return w, nil
}

// allocGrokState allocates all runtime buffers
func allocGrokState(cfg *GrokConfig) GrokState {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return GrokState{
		X:             make([]float32, cfg.EmbedDim),
		XB:            make([]float32, cfg.EmbedDim),
		XB2:           make([]float32, cfg.EmbedDim),
		XB3:           make([]float32, cfg.EmbedDim),
		HB:            make([]float32, cfg.IntermSize),
		HB2:           make([]float32, cfg.IntermSize),
		HBShared:      make([]float32, cfg.IntermSize),
		HB2Shared:     make([]float32, cfg.IntermSize),
		Q:             make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:             make([]float32, kvDim),
		V:             make([]float32, kvDim),
		Att:           make([]float32, cfg.NumHeads*cfg.SeqLen),
		Logits:        make([]float32, cfg.VocabSize),
		KeyCache:      make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		ValueCache:    make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		CosCache:      make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		SinCache:      make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		EmbBuf:        make([]float32, cfg.EmbedDim),
		RouterLogits:  make([]float32, cfg.NumExperts),
		ExpertOutputs: make([]float32, cfg.EmbedDim),
	}
}

func precomputeGrokRoPE(s *GrokState, cfg *GrokConfig) {
	half := cfg.HeadDim / 2
	theta := float64(cfg.RopeTheta)
	for pos := 0; pos < cfg.SeqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(cfg.HeadDim))
			angle := float64(pos) * freq
			s.CosCache[pos*half+i] = float32(math.Cos(angle))
			s.SinCache[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// gelu computes GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
func gelu(x float32) float32 {
	x64 := float64(x)
	return float32(0.5 * x64 * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x64+0.044715*x64*x64*x64))))
}

// softmaxClamp applies soft clamping: clamp * tanh(x / clamp) for each element
func softmaxClamp(att []float32, clamp float32) {
	if clamp <= 0 {
		return
	}
	invClamp := 1.0 / clamp
	for i := range att {
		att[i] = clamp * float32(math.Tanh(float64(att[i]*invClamp)))
	}
}

// topKExperts returns top-k expert indices and their softmax weights
func topKExperts(logits []float32, k int) ([]int, []float32) {
	type pair struct {
		idx int
		val float32
	}
	pairs := make([]pair, len(logits))
	for i, v := range logits {
		pairs[i] = pair{i, v}
	}
	sort.Slice(pairs, func(a, b int) bool { return pairs[a].val > pairs[b].val })

	topK := pairs[:k]
	indices := make([]int, k)
	weights := make([]float32, k)

	// Softmax over top-k logits
	var maxVal float32 = topK[0].val
	var sum float64
	for i, p := range topK {
		indices[i] = p.idx
		w := math.Exp(float64(p.val - maxVal))
		weights[i] = float32(w)
		sum += w
	}
	for i := range weights {
		weights[i] /= float32(sum)
	}
	return indices, weights
}

// Forward runs one token through the Grok MoE transformer
func (m *GrokModel) Forward(token int, pos int) {
	cfg := &m.Config
	w := &m.Weights
	s := &m.State
	dim := cfg.EmbedDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	hd := cfg.HeadDim
	headGroupSize := cfg.NumHeads / cfg.NumKVHeads

	// 1. Token embedding
	embedLookupInto(s.EmbBuf, w.TokenEmbed, w.TokenEmbType, token, dim)
	copy(s.X, s.EmbBuf)

	attnScale := float32(1.0 / math.Sqrt(float64(hd)))

	// 2. Transformer layers
	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// === ATTENTION ===
		// Pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V
		matmulDispatch(s.Q, l.WQ, l.WQType, s.XB, cfg.NumHeads*hd, dim)
		matmulDispatch(s.K, l.WK, l.WKType, s.XB, cfg.NumKVHeads*hd, dim)
		matmulDispatch(s.V, l.WV, l.WVType, s.XB, cfg.NumKVHeads*hd, dim)

		// RoPE
		for h := 0; h < cfg.NumHeads; h++ {
			applyRoPE(s.Q[h*hd:(h+1)*hd], pos, s.CosCache, s.SinCache, hd)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			applyRoPE(s.K[h*hd:(h+1)*hd], pos, s.CosCache, s.SinCache, hd)
		}

		// QK-norm (nanollama)
		if cfg.QKNorm {
			for h := 0; h < cfg.NumHeads; h++ {
				RMSNormBare(s.Q[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
			for h := 0; h < cfg.NumKVHeads; h++ {
				RMSNormBare(s.K[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
		}

		// KV cache
		cacheOff := layer*cfg.SeqLen*kvDim + pos*kvDim
		copy(s.KeyCache[cacheOff:cacheOff+kvDim], s.K[:kvDim])
		copy(s.ValueCache[cacheOff:cacheOff+kvDim], s.V[:kvDim])

		// Multi-head attention with GQA
		for h := 0; h < cfg.NumHeads; h++ {
			kvh := h / headGroupSize
			qh := s.Q[h*hd : (h+1)*hd]
			att := s.Att[h*cfg.SeqLen : h*cfg.SeqLen+pos+1]

			for t := 0; t <= pos; t++ {
				kOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				var dot float32
				for d := 0; d < hd; d++ {
					dot += qh[d] * s.KeyCache[kOff+d]
				}
				att[t] = dot * attnScale
			}

			// Grok attention clamping: clamp * tanh(logits / clamp)
			softmaxClamp(att[:pos+1], cfg.AttnClamp)

			Softmax(att, pos+1)

			xbSlice := s.XB2[h*hd : (h+1)*hd]
			for d := 0; d < hd; d++ {
				xbSlice[d] = 0
			}
			for t := 0; t <= pos; t++ {
				a := att[t]
				vOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				for d := 0; d < hd; d++ {
					xbSlice[d] += a * s.ValueCache[vOff+d]
				}
			}
		}

		// Output projection
		matmulDispatch(s.XB, l.WO, l.WOType, s.XB2, dim, dim)

		// Double pre-norm: post-norm on attention output
		if cfg.DoublePrenorm && l.AttnPostNorm != nil {
			RMSNormInto(s.XB3, s.XB, l.AttnPostNorm, cfg.RMSNormEps)
			copy(s.XB, s.XB3)
		}

		// Residual
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// === MoE FFN ===
		// Pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// Clear expert output accumulator
		for i := 0; i < dim; i++ {
			s.ExpertOutputs[i] = 0
		}

		if cfg.NumExperts > 1 && l.WRouter != nil {
			// Router: compute expert logits
			matmulDispatch(s.RouterLogits, l.WRouter, l.WRouterType, s.XB, cfg.NumExperts, dim)

			// Top-k expert selection with softmax weights
			expertIdx, expertWeights := topKExperts(s.RouterLogits[:cfg.NumExperts], cfg.NumExpertsPerTok)

			// Compute selected expert outputs
			for k := 0; k < cfg.NumExpertsPerTok; k++ {
				eIdx := expertIdx[k]
				eW := expertWeights[k]
				exp := &l.Experts[eIdx]

				// gate_proj + up_proj
				matmulDispatch(s.HB, exp.WGate, exp.WGateType, s.XB, cfg.IntermSize, dim)
				matmulDispatch(s.HB2, exp.WUp, exp.WUpType, s.XB, cfg.IntermSize, dim)

				// activation(gate) * up
				if cfg.UseGELU {
					for i := 0; i < cfg.IntermSize; i++ {
						s.HB[i] = gelu(s.HB[i]) * s.HB2[i]
					}
				} else {
					for i := 0; i < cfg.IntermSize; i++ {
						s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
					}
				}

				// down_proj → accumulate with expert weight
				matmulDispatch(s.XB2, exp.WDown, exp.WDownType, s.HB, dim, cfg.IntermSize)
				for i := 0; i < dim; i++ {
					s.ExpertOutputs[i] += eW * s.XB2[i]
				}
			}
		} else {
			// Dense fallback (single expert, no routing)
			exp := &l.Experts[0]
			matmulDispatch(s.HB, exp.WGate, exp.WGateType, s.XB, cfg.IntermSize, dim)
			matmulDispatch(s.HB2, exp.WUp, exp.WUpType, s.XB, cfg.IntermSize, dim)
			if cfg.UseGELU {
				for i := 0; i < cfg.IntermSize; i++ {
					s.HB[i] = gelu(s.HB[i]) * s.HB2[i]
				}
			} else {
				for i := 0; i < cfg.IntermSize; i++ {
					s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
				}
			}
			matmulDispatch(s.ExpertOutputs, exp.WDown, exp.WDownType, s.HB, dim, cfg.IntermSize)
		}

		// Shared expert (always-on, output added to MoE output)
		if l.SharedExpert != nil {
			se := l.SharedExpert
			matmulDispatch(s.HBShared, se.WGate, se.WGateType, s.XB, cfg.IntermSize, dim)
			matmulDispatch(s.HB2Shared, se.WUp, se.WUpType, s.XB, cfg.IntermSize, dim)
			if cfg.UseGELU {
				for i := 0; i < cfg.IntermSize; i++ {
					s.HBShared[i] = gelu(s.HBShared[i]) * s.HB2Shared[i]
				}
			} else {
				for i := 0; i < cfg.IntermSize; i++ {
					s.HBShared[i] = SiLU(s.HBShared[i]) * s.HB2Shared[i]
				}
			}
			matmulDispatch(s.XB2, se.WDown, se.WDownType, s.HBShared, dim, cfg.IntermSize)
			for i := 0; i < dim; i++ {
				s.ExpertOutputs[i] += s.XB2[i]
			}
		}

		// Double pre-norm: post-norm on FFN output
		if cfg.DoublePrenorm && l.FFNPostNorm != nil {
			RMSNormInto(s.XB3, s.ExpertOutputs, l.FFNPostNorm, cfg.RMSNormEps)
			copy(s.ExpertOutputs, s.XB3)
		}

		// Residual
		for i := 0; i < dim; i++ {
			s.X[i] += s.ExpertOutputs[i]
		}
	}

	// 3. Final norm
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)

	// 4. LM head → logits
	matmulDispatch(s.Logits, w.Output, w.OutputType, s.X, cfg.VocabSize, dim)

	// Output logit clamping (Grok-1 style, applied to final logits)
	if cfg.AttnClamp > 0 {
		softmaxClamp(s.Logits[:cfg.VocabSize], cfg.AttnClamp)
	}
}

// Reset clears KV cache and position
func (m *GrokModel) Reset() {
	for i := range m.State.KeyCache {
		m.State.KeyCache[i] = 0
	}
	for i := range m.State.ValueCache {
		m.State.ValueCache[i] = 0
	}
	m.State.Pos = 0
}
