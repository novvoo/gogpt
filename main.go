package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// ==================== 自动微分引擎 ====================

// Value 支持自动微分的标量值
type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	Op         string
	BackwardFn func()
}

func NewValue(data float64, children ...*Value) *Value {
	return &Value{
		Data:     data,
		Grad:     0,
		Children: children,
		Op:       "",
	}
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.Data+other.Data, v, other)
	out.Op = "+"
	out.BackwardFn = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.Data*other.Data, v, other)
	out.Op = "*"
	out.BackwardFn = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) Pow(exp float64) *Value {
	out := NewValue(math.Pow(v.Data, exp), v)
	out.Op = fmt.Sprintf("**%v", exp)
	out.BackwardFn = func() {
		v.Grad += exp * math.Pow(v.Data, exp-1) * out.Grad
	}
	return out
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1))
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) Exp() *Value {
	out := NewValue(math.Exp(v.Data), v)
	out.Op = "exp"
	out.BackwardFn = func() {
		v.Grad += out.Data * out.Grad
	}
	return out
}

func (v *Value) Log() *Value {
	out := NewValue(math.Log(v.Data), v)
	out.Op = "log"
	out.BackwardFn = func() {
		v.Grad += (1.0 / v.Data) * out.Grad
	}
	return out
}

func (v *Value) Relu() *Value {
	var outVal float64
	if v.Data > 0 {
		outVal = v.Data
	}
	out := NewValue(outVal, v)
	out.Op = "relu"
	out.BackwardFn = func() {
		if v.Data > 0 {
			v.Grad += out.Grad
		}
	}
	return out
}

// Backward 反向传播
func (v *Value) Backward() {
	// 拓扑排序
	topo := []*Value{}
	visited := make(map[*Value]bool)
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.Children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}
	buildTopo(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].BackwardFn != nil {
			topo[i].BackwardFn()
		}
	}
}

// ==================== 神经网络工具函数 ====================

// 矩阵乘法
func matMul(x []*Value, w [][]*Value) []*Value {
	nout := len(w)
	nin := len(x)
	out := make([]*Value, nout)
	for i := 0; i < nout; i++ {
		sum := NewValue(0)
		for j := 0; j < nin; j++ {
			sum = sum.Add(x[j].Mul(w[i][j]))
		}
		out[i] = sum
	}
	return out
}

// RMSNorm 归一化
func rmsnorm(x []*Value) []*Value {
	ms := NewValue(0)
	for _, xi := range x {
		ms = ms.Add(xi.Mul(xi))
	}
	ms = ms.Div(NewValue(float64(len(x))))
	scale := ms.Add(NewValue(1e-5)).Pow(-0.5)

	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = xi.Mul(scale)
	}
	return out
}

// Softmax
func softmax(x []*Value) []*Value {
	maxVal := x[0].Data
	for _, v := range x {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}

	expSum := NewValue(0)
	exps := make([]*Value, len(x))
	for i, v := range x {
		exps[i] = v.Sub(NewValue(maxVal)).Exp()
		expSum = expSum.Add(exps[i])
	}

	out := make([]*Value, len(x))
	for i, e := range exps {
		out[i] = e.Div(expSum)
	}
	return out
}

// 生成随机矩阵
func randMatrix(nout, nin int, std float64) [][]*Value {
	mat := make([][]*Value, nout)
	for i := 0; i < nout; i++ {
		row := make([]*Value, nin)
		for j := 0; j < nin; j++ {
			row[j] = NewValue(rand.NormFloat64() * std)
		}
		mat[i] = row
	}
	return mat
}

// ==================== GPT 模型 ====================

type GPTConfig struct {
	VocabSize int
	NEmbd     int
	NLayer    int
	NHead     int
	BlockSize int
}

type GPT struct {
	Config    GPTConfig
	StateDict map[string]interface{}
	Params    []*Value
}

func NewGPT(config GPTConfig) *GPT {
	gpt := &GPT{
		Config:    config,
		StateDict: make(map[string]interface{}),
		Params:    []*Value{},
	}

	std := 0.08
	// 嵌入层
	gpt.StateDict["wte"] = randMatrix(config.VocabSize, config.NEmbd, std)
	gpt.StateDict["wpe"] = randMatrix(config.BlockSize, config.NEmbd, std)
	gpt.StateDict["lm_head"] = randMatrix(config.VocabSize, config.NEmbd, std)

	// Transformer 层
	for i := 0; i < config.NLayer; i++ {
		prefix := fmt.Sprintf("layer%d", i)
		gpt.StateDict[prefix+".attn_wq"] = randMatrix(config.NEmbd, config.NEmbd, std)
		gpt.StateDict[prefix+".attn_wk"] = randMatrix(config.NEmbd, config.NEmbd, std)
		gpt.StateDict[prefix+".attn_wv"] = randMatrix(config.NEmbd, config.NEmbd, std)
		gpt.StateDict[prefix+".attn_wo"] = randMatrix(config.NEmbd, config.NEmbd, std)
		gpt.StateDict[prefix+".mlp_fc1"] = randMatrix(4*config.NEmbd, config.NEmbd, std)
		gpt.StateDict[prefix+".mlp_fc2"] = randMatrix(config.NEmbd, 4*config.NEmbd, std)
	}

	// 收集所有参数
	gpt.collectParams()
	return gpt
}

func (g *GPT) collectParams() {
	for _, v := range g.StateDict {
		switch val := v.(type) {
		case [][]*Value:
			for _, row := range val {
				g.Params = append(g.Params, row...)
			}
		}
	}
}

func (g *GPT) Forward(tokenID, posID int, keys, values [][][]*Value) []*Value {
	config := g.Config
	headDim := config.NEmbd / config.NHead

	// 嵌入
	wte := g.StateDict["wte"].([][]*Value)
	wpe := g.StateDict["wpe"].([][]*Value)

	tokEmb := wte[tokenID]
	posEmb := wpe[posID]

	x := make([]*Value, config.NEmbd)
	for i := 0; i < config.NEmbd; i++ {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = rmsnorm(x)

	// Transformer 层
	for li := 0; li < config.NLayer; li++ {
		prefix := fmt.Sprintf("layer%d", li)

		// 多头注意力
		xResidual := make([]*Value, len(x))
		copy(xResidual, x)
		x = rmsnorm(x)

		wq := g.StateDict[prefix+".attn_wq"].([][]*Value)
		wk := g.StateDict[prefix+".attn_wk"].([][]*Value)
		wv := g.StateDict[prefix+".attn_wv"].([][]*Value)
		wo := g.StateDict[prefix+".attn_wo"].([][]*Value)

		q := matMul(x, wq)
		k := matMul(x, wk)
		v := matMul(x, wv)

		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		// 多头计算
		xAttn := make([]*Value, config.NEmbd)
		for i := range xAttn {
			xAttn[i] = NewValue(0)
		}
		for h := 0; h < config.NHead; h++ {
			hs := h * headDim
			qh := q[hs : hs+headDim]

			// 计算注意力分数
			attnLogits := make([]*Value, len(keys[li]))
			for t := 0; t < len(keys[li]); t++ {
				kh := keys[li][t][hs : hs+headDim]
				sum := NewValue(0)
				for j := 0; j < headDim; j++ {
					sum = sum.Add(qh[j].Mul(kh[j]))
				}
				attnLogits[t] = sum.Div(NewValue(math.Sqrt(float64(headDim))))
			}

			attnWeights := softmax(attnLogits)

			// 加权求和
			for j := 0; j < headDim; j++ {
				sum := NewValue(0)
				for t := 0; t < len(values[li]); t++ {
					vh := values[li][t][hs : hs+headDim]
					sum = sum.Add(attnWeights[t].Mul(vh[j]))
				}
				xAttn[hs+j] = sum
			}
		}

		x = matMul(xAttn, wo)
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}

		// MLP
		xResidual = make([]*Value, len(x))
		copy(xResidual, x)
		x = rmsnorm(x)

		fc1 := g.StateDict[prefix+".mlp_fc1"].([][]*Value)
		fc2 := g.StateDict[prefix+".mlp_fc2"].([][]*Value)

		x = matMul(x, fc1)
		for i := range x {
			x[i] = x[i].Relu()
		}
		x = matMul(x, fc2)

		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}
	}

	// 输出头
	lmHead := g.StateDict["lm_head"].([][]*Value)
	logits := matMul(x, lmHead)
	return logits
}

// ==================== 训练管理 ====================

type TrainingState struct {
	sync.RWMutex
	IsTraining bool
	Step       int
	TotalSteps int
	Loss       float64
	Dataset    []string
	Vocab      []string
	VocabSize  int
	BOS        int
	Model      *GPT
	ModelPath  string
	History    []float64 // 损失历史
}

var trainingState = &TrainingState{
	History: make([]float64, 0),
}

type TrainRequest struct {
	NEmbd        int     `json:"n_embd"`
	NLayer       int     `json:"n_layer"`
	NHead        int     `json:"n_head"`
	BlockSize    int     `json:"block_size"`
	NumSteps     int     `json:"num_steps"`
	LearningRate float64 `json:"learning_rate"`
}

func validateTrainRequest(req TrainRequest) error {
	var errs []string
	if req.NEmbd <= 0 {
		errs = append(errs, "n_embd 必须为正")
	}
	if req.NLayer <= 0 {
		errs = append(errs, "n_layer 必须为正")
	}
	if req.NHead <= 0 {
		errs = append(errs, "n_head 必须为正")
	}
	if req.NEmbd > 0 && req.NHead > 0 {
		if req.NHead > req.NEmbd {
			errs = append(errs, "n_head 不得大于 n_embd")
		}
		if req.NEmbd%req.NHead != 0 {
			errs = append(errs, "n_embd 必须能被 n_head 整除")
		}
	}
	if req.BlockSize < 2 || req.BlockSize > 1024 {
		errs = append(errs, "block_size 需在 2..1024 内")
	}
	if req.NumSteps < 1 || req.NumSteps > 10000 {
		errs = append(errs, "num_steps 需在 1..10000 内")
	}
	if req.LearningRate < 1e-6 || req.LearningRate > 0.1 {
		errs = append(errs, "learning_rate 需在 [1e-6..0.1] 内")
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// Adam 优化器状态
type AdamState struct {
	M []float64
	V []float64
	T int
}

var adamState *AdamState

const defaultModelPath = "models/latest.json"

func initAdam(params []*Value) {
	adamState = &AdamState{
		M: make([]float64, len(params)),
		V: make([]float64, len(params)),
		T: 0,
	}
}

func adamStep(params []*Value, lr, beta1, beta2, eps float64, step, totalSteps int) {
	adamState.T++
	lrT := lr * (1 - float64(step)/float64(totalSteps))

	for i, p := range params {
		adamState.M[i] = beta1*adamState.M[i] + (1-beta1)*p.Grad
		adamState.V[i] = beta2*adamState.V[i] + (1-beta2)*p.Grad*p.Grad

		mHat := adamState.M[i] / (1 - math.Pow(beta1, float64(adamState.T)))
		vHat := adamState.V[i] / (1 - math.Pow(beta2, float64(adamState.T)))

		p.Data -= lrT * mHat / (math.Sqrt(vHat) + eps)
		p.Grad = 0
	}
}

// ==================== HTTP 处理 ====================

func main() {
	if err := loadModelFromDisk(defaultModelPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		panic(err)
	}

	r := gin.Default()

	// 静态文件和模板
	r.Static("/static", "./static")
	r.LoadHTMLGlob("templates/*")

	// 页面路由
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{
			"isTraining": trainingState.IsTraining,
		})
	})

	// API 路由
	api := r.Group("/api")
	{
		api.POST("/upload", handleUpload)
		api.POST("/train", handleTrain)
		api.POST("/stop", handleStop)
		api.GET("/status", handleStatus)
		api.POST("/generate", handleGenerate)
		api.GET("/stream", handleStream)
		api.POST("/model/import", handleModelImport)
		api.GET("/model/export", handleModelExport)
		api.POST("/model/save", handleModelSave)
	}

	if err := r.Run(":8080"); err != nil {
		panic(err)
	}
}

type ModelFile struct {
	Version    int                    `json:"version"`
	CreatedAt  string                 `json:"created_at"`
	Config     GPTConfig              `json:"config"`
	Vocab      []string               `json:"vocab"`
	BOS        int                    `json:"bos"`
	Step       int                    `json:"step"`
	TotalSteps int                    `json:"total_steps"`
	History    []float64              `json:"history"`
	Weights    map[string][][]float64 `json:"weights"`
}

func snapshotModelFile() (*ModelFile, error) {
	trainingState.RLock()
	model := trainingState.Model
	vocab := append([]string(nil), trainingState.Vocab...)
	bos := trainingState.BOS
	step := trainingState.Step
	totalSteps := trainingState.TotalSteps
	history := append([]float64(nil), trainingState.History...)
	trainingState.RUnlock()

	if model == nil {
		return nil, errors.New("model not loaded")
	}

	weights := make(map[string][][]float64, len(model.StateDict))
	for k, v := range model.StateDict {
		mat, ok := v.([][]*Value)
		if !ok {
			continue
		}
		out := make([][]float64, len(mat))
		for i := range mat {
			out[i] = make([]float64, len(mat[i]))
			for j := range mat[i] {
				out[i][j] = mat[i][j].Data
			}
		}
		weights[k] = out
	}

	return &ModelFile{
		Version:    1,
		CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
		Config:     model.Config,
		Vocab:      vocab,
		BOS:        bos,
		Step:       step,
		TotalSteps: totalSteps,
		History:    history,
		Weights:    weights,
	}, nil
}

func applyModelFile(mf *ModelFile) error {
	if mf == nil {
		return errors.New("nil model file")
	}
	if mf.Version != 1 {
		return fmt.Errorf("unsupported model version: %d", mf.Version)
	}
	if mf.Config.VocabSize <= 0 || mf.Config.NEmbd <= 0 || mf.Config.NLayer <= 0 || mf.Config.NHead <= 0 || mf.Config.BlockSize <= 0 {
		return errors.New("invalid config")
	}
	if mf.Config.NEmbd%mf.Config.NHead != 0 {
		return errors.New("invalid config: n_embd must be divisible by n_head")
	}
	if len(mf.Vocab)+1 != mf.Config.VocabSize {
		return errors.New("vocab size mismatch with config")
	}
	if mf.BOS != len(mf.Vocab) {
		return errors.New("invalid BOS id")
	}

	model := NewGPT(mf.Config)

	for k, v := range model.StateDict {
		mat, ok := v.([][]*Value)
		if !ok {
			continue
		}
		src, ok := mf.Weights[k]
		if !ok {
			return fmt.Errorf("missing weights key: %s", k)
		}
		if len(src) != len(mat) {
			return fmt.Errorf("weights shape mismatch for %s", k)
		}
		for i := range mat {
			if len(src[i]) != len(mat[i]) {
				return fmt.Errorf("weights shape mismatch for %s", k)
			}
			for j := range mat[i] {
				mat[i][j].Data = src[i][j]
				mat[i][j].Grad = 0
			}
		}
	}

	trainingState.Lock()
	trainingState.Model = model
	trainingState.Vocab = append([]string(nil), mf.Vocab...)
	trainingState.BOS = mf.BOS
	trainingState.VocabSize = mf.Config.VocabSize
	trainingState.Step = mf.Step
	trainingState.TotalSteps = mf.TotalSteps
	trainingState.History = append([]float64(nil), mf.History...)
	trainingState.IsTraining = false
	trainingState.Unlock()

	return nil
}

func saveModelToDisk(path string) error {
	mf, err := snapshotModelFile()
	if err != nil {
		return err
	}

	b, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, "model-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return err
	}

	trainingState.Lock()
	trainingState.ModelPath = path
	trainingState.Unlock()
	return nil
}

func loadModelFromDisk(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var mf ModelFile
	if err := json.Unmarshal(b, &mf); err != nil {
		return err
	}
	if err := applyModelFile(&mf); err != nil {
		return err
	}
	trainingState.Lock()
	trainingState.ModelPath = path
	trainingState.Unlock()
	return nil
}

func handleUpload(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(400, gin.H{"error": "No file uploaded"})
		return
	}

	const maxUploadSize int64 = 10 << 20 // 10 MiB
	if file.Size > maxUploadSize {
		c.JSON(400, gin.H{"error": "File too large"})
		return
	}

	// 打开文件
	f, err := file.Open()
	if err != nil {
		c.JSON(500, gin.H{"error": "Cannot open file"})
		return
	}
	defer f.Close()

	// MIME 检测
	header := make([]byte, 512)
	n, _ := f.Read(header)
	contentType := http.DetectContentType(header[:n])

	// 简单的白名单检查
	isText := strings.HasPrefix(contentType, "text/") ||
		contentType == "application/json" ||
		contentType == "application/xml"

	if !isText {
		c.JSON(400, gin.H{"error": "Unsupported file type: " + contentType})
		return
	}

	// 重置文件指针
	if _, err := f.Seek(0, 0); err != nil {
		c.JSON(500, gin.H{"error": "Failed to reset file pointer"})
		return
	}

	// 流式读取
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 16<<20) // 1MB init, 16MB max

	var docs []string
	charSet := make(map[string]bool)
	var totalRead int64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		totalRead += int64(len(line)) + 1
		if totalRead > maxUploadSize {
			c.JSON(400, gin.H{"error": "File content too large"})
			return
		}

		if line != "" {
			docs = append(docs, line)
			for _, ch := range line {
				charSet[string(ch)] = true
			}
		}
	}

	if err := scanner.Err(); err != nil {
		c.JSON(400, gin.H{"error": "Read error: " + err.Error()})
		return
	}

	// 构建词表
	vocab := []string{}
	for ch := range charSet {
		vocab = append(vocab, ch)
	}
	sort.Strings(vocab)

	trainingState.Lock()
	defer trainingState.Unlock()

	trainingState.Dataset = docs
	trainingState.Vocab = vocab
	trainingState.VocabSize = len(vocab) + 1 // +1 for BOS
	trainingState.BOS = len(vocab)
	trainingState.Step = 0
	trainingState.History = []float64{}

	c.JSON(200, gin.H{
		"message":   "Upload successful",
		"samples":   len(docs),
		"vocabSize": trainingState.VocabSize,
		"vocab":     vocab,
	})
}

func handleTrain(c *gin.Context) {
	trainingState.Lock()
	if trainingState.IsTraining {
		trainingState.Unlock()
		c.JSON(400, gin.H{"error": "Already training"})
		return
	}
	if len(trainingState.Dataset) == 0 {
		trainingState.Unlock()
		c.JSON(400, gin.H{"error": "No dataset uploaded"})
		return
	}

	var req TrainRequest

	if err := c.BindJSON(&req); err != nil {
		trainingState.Unlock()
		c.JSON(400, gin.H{"error": "Invalid parameters"})
		return
	}
	if err := validateTrainRequest(req); err != nil {
		trainingState.Unlock()
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	rand.Seed(42)
	rand.Shuffle(len(trainingState.Dataset), func(i, j int) {
		trainingState.Dataset[i], trainingState.Dataset[j] = trainingState.Dataset[j], trainingState.Dataset[i]
	})

	// 初始化模型
	config := GPTConfig{
		VocabSize: trainingState.VocabSize,
		NEmbd:     req.NEmbd,
		NLayer:    req.NLayer,
		NHead:     req.NHead,
		BlockSize: req.BlockSize,
	}
	trainingState.Model = NewGPT(config)
	trainingState.TotalSteps = req.NumSteps
	trainingState.IsTraining = true
	initAdam(trainingState.Model.Params)
	trainingState.Unlock()

	// 后台训练
	go trainLoop(req.LearningRate)

	c.JSON(200, gin.H{"message": "Training started"})
}

func trainLoop(learningRate float64) {
	trainingState.RLock()
	dataset := trainingState.Dataset
	vocab := trainingState.Vocab
	bos := trainingState.BOS
	config := trainingState.Model.Config
	model := trainingState.Model
	totalSteps := trainingState.TotalSteps
	trainingState.RUnlock()

	// 字符到ID的映射
	charToID := make(map[string]int)
	for i, ch := range vocab {
		charToID[ch] = i
	}

	for step := 0; step < totalSteps; step++ {
		trainingState.RLock()
		if !trainingState.IsTraining {
			trainingState.RUnlock()
			break
		}
		trainingState.RUnlock()

		doc := dataset[step%len(dataset)]

		// Tokenize: BOS + chars + BOS
		tokens := []int{bos}
		for _, ch := range doc {
			if id, ok := charToID[string(ch)]; ok {
				tokens = append(tokens, id)
			}
		}
		tokens = append(tokens, bos)

		n := len(tokens) - 1
		if n > config.BlockSize {
			n = config.BlockSize
		}

		// 前向传播
		keys := make([][][]*Value, config.NLayer)
		values := make([][][]*Value, config.NLayer)

		losses := []*Value{}
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]

			logits := model.Forward(tokenID, posID, keys, values)
			probs := softmax(logits)
			// Cross Entropy Loss: -log(p)
			// probs[targetID] is the probability of the correct token (0-1)
			// Log(p) is negative, so we negate it to get positive loss
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}

		// 计算平均损失
		loss := NewValue(0)
		for _, l := range losses {
			loss = loss.Add(l)
		}
		loss = loss.Div(NewValue(float64(n)))

		// 反向传播
		loss.Backward()

		// 更新参数
		adamStep(model.Params, learningRate, 0.85, 0.99, 1e-8, step, totalSteps)

		// 更新状态
		trainingState.Lock()
		trainingState.Step = step + 1

		// 简单的梯度裁剪/数值稳定
		currentLoss := loss.Data
		if math.IsNaN(currentLoss) || math.IsInf(currentLoss, 0) {
			currentLoss = 100.0 // 设定一个较大的惩罚值，或者保持上一步的 loss
			if len(trainingState.History) > 0 {
				currentLoss = trainingState.History[len(trainingState.History)-1]
			}
		}

		trainingState.Loss = currentLoss
		trainingState.History = append(trainingState.History, currentLoss)
		trainingState.Unlock()
	}

	trainingState.RLock()
	step := trainingState.Step
	totalSteps = trainingState.TotalSteps
	trainingState.RUnlock()

	trainingState.Lock()
	trainingState.IsTraining = false
	trainingState.Unlock()

	if step >= totalSteps && totalSteps > 0 {
		_ = saveModelToDisk(defaultModelPath)
	}
}

func handleStop(c *gin.Context) {
	trainingState.Lock()
	trainingState.IsTraining = false
	trainingState.Unlock()
	c.JSON(200, gin.H{"message": "Training stopped"})
}

func handleStatus(c *gin.Context) {
	trainingState.RLock()
	defer trainingState.RUnlock()

	// Handle NaN/Inf in Loss
	currentLoss := trainingState.Loss
	if math.IsNaN(currentLoss) || math.IsInf(currentLoss, 0) {
		currentLoss = 0
	}

	// Handle NaN/Inf in History
	// Create a copy to avoid race conditions if we were modifying in place (though we are under RLock)
	// and to sanitize the data for JSON
	history := make([]float64, len(trainingState.History))
	for i, v := range trainingState.History {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			history[i] = 0
		} else {
			history[i] = v
		}
	}

	c.JSON(200, gin.H{
		"isTraining":  trainingState.IsTraining,
		"step":        trainingState.Step,
		"totalSteps":  trainingState.TotalSteps,
		"loss":        currentLoss,
		"history":     history,
		"datasetSize": len(trainingState.Dataset),
		"modelLoaded": trainingState.Model != nil,
		"modelPath":   trainingState.ModelPath,
	})
}

func handleGenerate(c *gin.Context) {
	trainingState.RLock()
	if trainingState.Model == nil {
		trainingState.RUnlock()
		c.JSON(400, gin.H{"error": "Model not trained"})
		return
	}
	model := trainingState.Model
	vocab := trainingState.Vocab
	bos := trainingState.BOS
	config := trainingState.Model.Config
	trainingState.RUnlock()

	idToChar := make(map[int]string)
	for i, ch := range vocab {
		idToChar[i] = ch
	}
	idToChar[bos] = "<BOS>"

	var req struct {
		Prompt string `json:"prompt"`
		MaxLen int    `json:"max_len"`
	}

	if err := c.BindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}

	// 编码 prompt
	charToID := make(map[string]int)
	for i, ch := range vocab {
		charToID[ch] = i
	}

	tokens := []int{bos}
	for _, ch := range req.Prompt {
		if id, ok := charToID[string(ch)]; ok {
			tokens = append(tokens, id)
		}
	}

	// 生成
	keys := make([][][]*Value, config.NLayer)
	values := make([][][]*Value, config.NLayer)

	// 先处理已有 token
	for posID, tokenID := range tokens {
		_ = model.Forward(tokenID, posID, keys, values)
	}

	// 生成新 token
	generated := req.Prompt
	for i := 0; i < req.MaxLen && len(tokens) < config.BlockSize; i++ {
		posID := len(tokens) - 1
		tokenID := tokens[len(tokens)-1]

		logits := model.Forward(tokenID, posID, keys, values)
		probs := softmax(logits)

		// 采样
		r := rand.Float64()
		cumSum := 0.0
		nextID := bos
		for id, p := range probs {
			cumSum += p.Data
			if r < cumSum {
				nextID = id
				break
			}
		}

		if nextID == bos {
			break
		}

		tokens = append(tokens, nextID)
		generated += idToChar[nextID]
	}

	c.JSON(200, gin.H{
		"generated": generated,
		"tokens":    len(tokens),
	})
}

func handleStream(c *gin.Context) {
	c.Stream(func(w io.Writer) bool {
		trainingState.RLock()
		loss := trainingState.Loss
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			loss = 0
		}
		data := gin.H{
			"isTraining": trainingState.IsTraining,
			"step":       trainingState.Step,
			"totalSteps": trainingState.TotalSteps,
			"loss":       loss,
		}
		trainingState.RUnlock()

		json.NewEncoder(w).Encode(data)
		return false
	})
}

func handleModelSave(c *gin.Context) {
	var req struct {
		Name string `json:"name"`
	}
	_ = c.BindJSON(&req)

	path := defaultModelPath
	if req.Name != "" {
		name := filepath.Base(req.Name)
		if !strings.HasSuffix(strings.ToLower(name), ".json") {
			name += ".json"
		}
		path = filepath.Join("models", name)
	}

	if err := saveModelToDisk(path); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"message": "Model saved", "path": path})
}

func handleModelExport(c *gin.Context) {
	path := defaultModelPath
	if b, err := os.ReadFile(path); err == nil {
		c.Header("Content-Type", "application/json")
		c.Header("Content-Disposition", `attachment; filename="model.json"`)
		c.Data(200, "application/json", b)
		return
	}

	mf, err := snapshotModelFile()
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	b, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to encode model"})
		return
	}
	c.Header("Content-Type", "application/json")
	c.Header("Content-Disposition", `attachment; filename="model.json"`)
	c.Data(200, "application/json", b)
}

func handleModelImport(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(400, gin.H{"error": "No file uploaded"})
		return
	}

	const maxUploadSize int64 = 50 << 20
	if file.Size > maxUploadSize {
		c.JSON(400, gin.H{"error": "File too large"})
		return
	}

	f, err := file.Open()
	if err != nil {
		c.JSON(500, gin.H{"error": "Cannot open file"})
		return
	}
	defer f.Close()

	b, err := io.ReadAll(io.LimitReader(f, maxUploadSize+1))
	if err != nil {
		c.JSON(400, gin.H{"error": "Read error: " + err.Error()})
		return
	}
	if int64(len(b)) > maxUploadSize {
		c.JSON(400, gin.H{"error": "File too large"})
		return
	}

	var mf ModelFile
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&mf); err != nil {
		c.JSON(400, gin.H{"error": "Invalid model file: " + err.Error()})
		return
	}
	if err := applyModelFile(&mf); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	_ = saveModelToDisk(defaultModelPath)
	c.JSON(200, gin.H{"message": "Model imported", "path": defaultModelPath})
}
