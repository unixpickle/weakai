package main

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"github.com/unixpickle/weakai/svm"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: face_recog <negatives> <positives> <best_vec>")
		os.Exit(1)
	}

	fmt.Println("Loading images...")
	negatives := loadImages(os.Args[1])
	positives := loadImages(os.Args[2])

	allImages := make([]*Image, len(negatives)+len(positives))
	copy(allImages, negatives)
	copy(allImages[len(negatives):], positives)
	width, height := minimumSize(allImages)

	problem := &svm.Problem{
		Kernel:    svm.CachedKernel(svm.LinearKernel),
		Positives: make([]svm.Sample, len(positives)),
		Negatives: make([]svm.Sample, len(negatives)),
	}

	fmt.Println("Cropping samples...")
	// TODO: perhaps normalize the image vectors.
	for i, x := range negatives {
		vec := cutOutMiddle(x, width, height).Vector
		problem.Negatives[i] = svm.Sample{V: vec, UserInfo: i + 1}
	}
	for i, x := range positives {
		vec := cutOutMiddle(x, width, height).Vector
		problem.Positives[i] = svm.Sample{V: vec, UserInfo: len(negatives) + 1 + i}
	}

	fmt.Println("Solving...")
	solver := svm.GradientDescentSolver{
		Tradeoff: 0.0001,
		Steps:    100000,
		StepSize: 0.001,
	}
	classifier := solver.Solve(problem).Linearize()

	image := imageForSolution(width, height, classifier.HyperplaneNormal.V)
	if err := image.WriteFile(os.Args[3]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func loadImages(dirPath string) []*Image {
	file, err := os.Open(dirPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer file.Close()
	names, err := file.Readdirnames(-1)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	res := make([]*Image, 0, len(names))
	for _, name := range names {
		if strings.HasPrefix(name, ".") {
			continue
		}
		image, err := ReadImage(filepath.Join(dirPath, name))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		res = append(res, image)
	}
	return res
}

func minimumSize(images []*Image) (width, height int) {
	width, height = images[0].Width, images[0].Height
	for _, img := range images {
		if img.Width < width {
			width = img.Width
		}
		if img.Height < height {
			height = img.Height
		}
	}
	return
}

func cutOutMiddle(img *Image, width, height int) *Image {
	cutLeft := (img.Width - width) / 2
	cutTop := (img.Height - height) / 2
	return img.Crop(image.Rect(cutLeft, cutTop, cutLeft+width, cutTop+height))
}

func imageForSolution(width, height int, solution []float64) *Image {
	var minPixel float64
	var maxPixel float64
	for i, sample := range solution {
		if sample < minPixel || i == 0 {
			minPixel = sample
		}
		if sample > maxPixel || i == 0 {
			maxPixel = sample
		}
	}
	for i := range solution {
		solution[i] += minPixel
		solution[i] /= (maxPixel - minPixel)
	}

	return &Image{
		Vector: solution,
		Width:  width,
		Height: height,
	}
}
