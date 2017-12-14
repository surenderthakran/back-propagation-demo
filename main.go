package main

import (
  "fmt"
  "math/rand"
  "strings"
  "time"
)

func main() {
  // setting timestamp as seed for random number generator.
  rand.Seed(time.Now().UnixNano())

  // training set containing the input/output sets for the AND gate.
  trainingSet := [][][]float64{
		[][]float64{[]float64{0, 0}, []float64{0}},
		[][]float64{[]float64{0, 1}, []float64{1}},
		[][]float64{[]float64{1, 0}, []float64{1}},
		[][]float64{[]float64{1, 1}, []float64{0}},
	}

  // initializing a new neural network with 2 nodes in the input layer (because
  // the AND gate accepts 2 inputs), 2 neurons in the hidden layer and 1 neuron
  // in the output layer (because the AND gate emits a single output value).
  mind, err := NewNeuralNetwork(2, 2, 1)
	if err != nil {
		fmt.Errorf("error in creating new neural network: %v", err)
		return
	}

  // we will iterate over the trainingSet where in each iteration we select a
  // random I/O pair from the set to train our netrowk.
  for i := 0; i < 10000; i++ {
		rand := rand.Intn(4)
		input := trainingSet[rand][0]
		output := trainingSet[rand][1]

		mind.Train(input, output)

    fmt.Printf("Input: %v, Target: %v, Actual: %v, Error: %v \n", input, output, mind.LastOutput(), mind.CalculateError(output))
	}

  fmt.Println(strings.Repeat("#", 100))
  // prints the structure and current set of weights in the network after training.
  mind.Describe()
}
