package main

import (
	"fmt"
	embedding2 "neural/sandbox/embedding"
	"strings"
)

func main() {

	sequenceLentgth := 10

	prompts := []string{"el pico es muy pelado", "la bandurria tiene un pico peludo", "la bandurria no es peluda"}

	vocabulary := embedding2.NewVocabulary()
	embeddings := embedding2.NewEmbeddings(4, sequenceLentgth, 5, 13)

	idBatches := make([][]int, len(prompts))
	for i, p := range prompts {
		ids := vocabulary.ResolveIds(strings.Split(p, " ")) // debería quedar sorted  cross-bacth
		idBatches[i] = ids
	}

	fmt.Printf("Dict:\n%v\n\n", idBatches)

	embeddingBatches := embeddings.BatchLookup(idBatches, sequenceLentgth)

	fmt.Printf("Embeddings:\n%v\n\n", embeddingBatches)

	vectors := embedding2.BuildEncodings(4, sequenceLentgth, 10000)

	fmt.Printf("Vectors:\n%v\n\n", vectors)

	output := make([][][]float64, len(embeddingBatches))
	for i, b := range embeddingBatches {
		output[i] = embedding2.AddEmbeddings(b, vectors, 4)
	}

	fmt.Printf("Output:\n%v\n\n", output)

	for i, b := range output {
		embedding2.Dropout(b, 4, 0.1, 7)
		output[i] = b
	}

}
