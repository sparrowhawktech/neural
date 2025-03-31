package main

import (
	"neural/backend"
	"neural/backend/simple/embeddings"
	"neural/backend/simple/tensors"
)

func main() {
	sequenceLen := 10
	dimension := 3
	num := 10000
	embeddingsSeed := int64(12)
	dropoutSeed := int64(7)
	dropoutP := 0.1

	tensorsApi := tensors.NewApi()
	simpleEmbeddingsApi := embeddings.NewEmbeddingsApi(tensorsApi, embeddingsSeed)
	embeddingsApi := backend.EmbeddingsApi(simpleEmbeddingsApi)

	vocabularyHandle := embeddingsApi.CreateVocabulary()

	prompts := []string{"el pico es muy pelado", "la bandurria tiene un pico peludo", "la bandurria no es peluda"}

	idBatches := make([][]int, len(prompts)) //change to tensors

	for i, p := range prompts {
		idBatches[i] = embeddingsApi.TokenizePrompt(vocabularyHandle, p)
	}

	embeddingsHandle := embeddingsApi.BuildEmbeddings(sequenceLen, dimension)
	embeddingBatchesHandle := embeddingsApi.EmbeddingsLookup(embeddingsHandle, idBatches)
	encodingsHandle := embeddingsApi.BuildEncodings(sequenceLen, dimension, num)
	outputHandle := embeddingsApi.AddEncodings(embeddingBatchesHandle, encodingsHandle)
	embeddingsApi.Dropout(outputHandle, dropoutP, dropoutSeed)

}
