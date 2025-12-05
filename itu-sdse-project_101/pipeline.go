package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("/project", client.Host().Directory("."))

	// Install dependencies
	python = python.WithExec([]string{"pip", "install", "-r", "/project/requirements.txt"})

	// DVC pull data
	python = python.WithExec([]string{"dvc", "pull", "/project/data/raw"})

	// Run preprocessing
	python = python.WithExec([]string{"python", "/project/itu_mlops_project_101/data_preprocessing.py"})

	// Run training
	python = python.WithExec([]string{"python", "/project/itu_mlops_project_101/model_training.py"})

	_, err = python.
		Directory("/project/models").
		Export(ctx, "models")
	if err != nil {
		return err
	}

	return nil
}
