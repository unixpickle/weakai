package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: constraints USA.svg <output_map.svg>")
		os.Exit(1)
	}

	variables := VariablesForStates()
	cancelChan := make(chan struct{})
	solutions := SolveConstraints(variables, cancelChan)

	solution := <-solutions
	close(cancelChan)

	cssCode := ""
	for state, color := range solution {
		stateAbbrev := state.(*StateVariable).Name
		cssCode += "#" + stateAbbrev + "{fill:" + color.(string) + ";}"
	}

	outputCode, err := setupOutputMap(cssCode)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not use input file:", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(os.Args[2], []byte(outputCode), 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func setupOutputMap(cssCode string) (string, error) {
	plainMap, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		return "", err
	}
	mapString := string(plainMap)
	return strings.Replace(mapString, "/* Insert CSS rules here */", cssCode, 1), nil
}
