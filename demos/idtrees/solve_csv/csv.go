package main

import (
	"encoding/csv"
	"errors"
	"io"
	"strconv"
	"strings"

	"github.com/unixpickle/weakai/idtrees"
)

func ReadCSV(r io.Reader) (samples []idtrees.Sample, keys []string, err error) {
	records, err := readStringRecords(r)
	if err != nil {
		return nil, nil, err
	}
	rawVals := stringsToValues(records)
	key, err := classKey(rawVals)
	if err != nil {
		return nil, nil, err
	}
	res := make([]idtrees.Sample, len(rawVals))
	for i, x := range rawVals {
		res[i] = &csvSample{
			Map:       x,
			ClassAttr: key,
		}
	}
	return res, trainingKeys(rawVals), nil
}

type csvSample struct {
	Map       map[string]interface{}
	ClassAttr string
}

func (c *csvSample) Attr(name string) interface{} {
	return c.Map[name]
}

func (c *csvSample) Class() interface{} {
	return c.Map[c.ClassAttr]
}

// trainingKeys returns the keys in the data set to
// use for training.
func trainingKeys(c []map[string]interface{}) []string {
	var res []string
	if len(c) > 0 {
		for key := range c[0] {
			if !strings.HasPrefix(key, "_") && !strings.HasPrefix(key, "*") {
				res = append(res, key)
			}
		}
	}
	return res
}

func classKey(c []map[string]interface{}) (string, error) {
	var useKey string
	if len(c) > 0 {
		for key := range c[0] {
			if strings.HasPrefix(key, "*") {
				if useKey != "" {
					return "", errors.New("multiple keys begin with asterisk")
				}
				useKey = key
			}
		}
	}
	if useKey == "" {
		return "", errors.New("missing class attribute " +
			"(name prefixed with asterisk)")
	}
	return useKey, nil
}

func readStringRecords(r io.Reader) ([]map[string]string, error) {
	csvReader := csv.NewReader(r)
	allRecords, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(allRecords) < 2 {
		return nil, nil
	}
	header := allRecords[0]
	res := make([]map[string]string, len(allRecords)-1)
	for i, x := range allRecords[1:] {
		r := map[string]string{}
		for j, val := range x {
			r[header[j]] = val
		}
		res[i] = r
	}
	return res, nil
}

func stringsToValues(strs []map[string]string) []map[string]interface{} {
	if len(strs) == 0 {
		return nil
	}
	res := make([]map[string]interface{}, len(strs))
	for i := range res {
		res[i] = map[string]interface{}{}
	}
	for key := range strs[0] {
		allInt, allFloat := true, true
		for _, x := range strs {
			_, intErr := strconv.ParseInt(x[key], 0, 64)
			_, floatErr := strconv.ParseFloat(x[key], 64)
			if intErr != nil {
				allInt = false
			}
			if floatErr != nil {
				allFloat = false
			}
		}
		if allInt {
			for i, x := range strs {
				num, _ := strconv.ParseInt(x[key], 0, 64)
				res[i][key] = num
			}
		} else if allFloat {
			for i, x := range strs {
				num, _ := strconv.ParseFloat(x[key], 64)
				res[i][key] = num
			}
		} else {
			for i, x := range strs {
				res[i][key] = x[key]
			}
		}
	}
	return res
}
