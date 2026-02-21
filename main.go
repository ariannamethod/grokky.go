package main

// main.go — Grok MoE inference engine
// Pure Go, no PyTorch. GGUF weights, MoE routing, quantized matmul.
// github.com/ariannamethod/grokky.go

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

func main() {
	weightsPath := flag.String("model", "", "path to GGUF weights file")
	maxTokens := flag.Int("n", 512, "max tokens to generate")
	temp := flag.Float64("temp", 0.6, "temperature")
	topP := flag.Float64("top-p", 0.95, "top-p nucleus sampling")
	prompt := flag.String("prompt", "", "single prompt (non-interactive)")
	flag.Parse()

	if *weightsPath == "" {
		fmt.Fprintln(os.Stderr, "usage: grokky -model <path.gguf> [-prompt 'text'] [-n 512] [-temp 0.6]")
		os.Exit(1)
	}

	fmt.Printf("[grokky] loading %s\n", *weightsPath)
	gguf, err := LoadGGUF(*weightsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading GGUF: %v\n", err)
		os.Exit(1)
	}

	model, err := LoadGrokModel(gguf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}

	tokenizer := NewTokenizer(&gguf.Meta)
	eosID := tokenizer.EosID

	fmt.Printf("[grokky] ready: %dL %dD %dH %dKV experts=%dx%d vocab=%d\n",
		model.Config.NumLayers, model.Config.EmbedDim, model.Config.NumHeads,
		model.Config.NumKVHeads, model.Config.NumExperts, model.Config.NumExpertsPerTok,
		model.Config.VocabSize)

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	generate := func(userPrompt string) {
		tokens := tokenizer.Encode(userPrompt, true)
		model.Reset()

		// Prefill
		prefillStart := time.Now()
		pos := 0
		for _, tok := range tokens {
			model.Forward(tok, pos)
			pos++
			if pos >= model.Config.SeqLen-1 {
				break
			}
		}
		prefillDur := time.Since(prefillStart)
		fmt.Printf("[grokky] prefill: %d tokens in %.2fs (%.1f tok/s)\n",
			len(tokens), prefillDur.Seconds(), float64(len(tokens))/prefillDur.Seconds())

		// Decode
		repWindow := 64
		recentTokens := make([]int, 0, repWindow)
		repPenalty := float32(1.15)
		decodeStart := time.Now()
		genTokens := 0

		for i := 0; i < *maxTokens; i++ {
			logits := model.State.Logits

			// Repetition penalty
			if repPenalty > 1.0 {
				for _, tok := range recentTokens {
					if tok >= 0 && tok < model.Config.VocabSize {
						if logits[tok] > 0 {
							logits[tok] /= repPenalty
						} else {
							logits[tok] *= repPenalty
						}
					}
				}
			}

			next := sampleTopP(logits, model.Config.VocabSize, float32(*temp), float32(*topP), rng)

			recentTokens = append(recentTokens, next)
			if len(recentTokens) > repWindow {
				recentTokens = recentTokens[1:]
			}

			if next == eosID {
				break
			}

			piece := tokenizer.DecodeToken(next)
			fmt.Print(piece)
			os.Stdout.Sync()

			model.Forward(next, pos)
			pos++
			genTokens++
			if pos >= model.Config.SeqLen {
				break
			}
		}
		decodeDur := time.Since(decodeStart)
		fmt.Printf("\n[grokky] decode: %d tokens in %.2fs (%.1f tok/s)\n",
			genTokens, decodeDur.Seconds(), float64(genTokens)/decodeDur.Seconds())
	}

	if *prompt != "" {
		generate(*prompt)
		return
	}

	// Interactive mode
	fmt.Println("[grokky] interactive mode. type 'quit' to exit.")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}
		generate(input)
	}
}

// sampleTopP does nucleus sampling
func sampleTopP(logits []float32, vocab int, temp, topP float32, rng *rand.Rand) int {
	if temp <= 0 {
		best := 0
		for i := 1; i < vocab; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	maxVal := logits[0]
	for i := 1; i < vocab; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	type idxProb struct {
		idx  int
		prob float32
	}
	candidates := make([]idxProb, 0, 1000)
	var sum float32
	for i := 0; i < vocab; i++ {
		p := float32(math.Exp(float64((logits[i] - maxVal) / temp)))
		if p > 1e-8 {
			candidates = append(candidates, idxProb{i, p})
			sum += p
		}
	}

	invSum := float32(1.0) / sum
	for i := range candidates {
		candidates[i].prob *= invSum
	}

	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].prob > candidates[i].prob {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
		var cum float32
		for k := 0; k <= i; k++ {
			cum += candidates[k].prob
		}
		if cum >= topP {
			break
		}
	}

	var cumsum float32
	r := rng.Float32()
	for _, c := range candidates {
		cumsum += c.prob
		if cumsum >= topP && cumsum-c.prob >= topP {
			break
		}
		if r <= cumsum {
			return c.idx
		}
	}
	if len(candidates) > 0 {
		return candidates[0].idx
	}
	return 0
}
