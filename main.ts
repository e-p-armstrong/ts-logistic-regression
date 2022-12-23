// typescript logistic regression package

// perform logistic regression and return a prediction function
function logistic (X, y, alpha=0.1,iterations=10000,predictionThreshold=0.5) {
    
    /*
        X is a 2d array of inputs
        y is a 1d array of answers
        alpha is the learning rate
        iterations is how many times to run regression
        W is the initial parameters of the model, 1d array
    */

    // normalize each value of X
    let [X_norm, maxes, mins, means]  = norm(X)


    let w = zeroes(X_norm[0]).map(() => Math.ceil(Math.random()))
    let b = Math.ceil(Math.random())

    // begin gradient descent
    for (let n = 0; n < iterations; n++) { // for n iterations
        // make a list of dw_djs for each w
        let dw_dj = costDerivativesLogistic(w,b,X_norm,y)
        let db_dj = biasDerivativeLogistic(w,b,X_norm,y)

        // then update each w with the calculated derivatives
        for (let j = 0; j < w.length; j++) {
            w[j] = w[j] - (alpha*dw_dj[j])
        }
        // update bias parameter
        b = b - (alpha * db_dj)
        if (n % 100 === 0) {
            console.log("w is", w, "and b is",b)
            console.log(`Cost is ${costLogistic(w,b,X_norm,y)}`)
        }
    }

    // prediction function
    return function (X_new) {
        let XNewScaled = scale(X_new,maxes,mins,means)
        if (predictionThreshold < g(w,b,XNewScaled)) {
            return 1
        } else {
            return 0
        }
    }
}

function costDerivativesLogistic (w,b,X,y) {
    /* 
       w is the 1d array of non-bias parameters
       b is the bias parameter
       X is the 2d array of inputs
       y is the 1d array of results
    */

    // for each w, make a derivative and add it to the
    // results array
    let resultArr = zeroes(w)
    
    for (let j = 0; j < resultArr.length; j ++) { // for each parameter

        // find the cost for this parameter
        let result = 0;
        for (let i = 0; i < X.length; i++) { // for each observation
            let loss = g(w,b,X[i]) - y[i]
            result += loss*X[i][j]
        }
        result = result/X.length
        
        resultArr[j] = result
    }
    return resultArr
}

test("bias derivative 1", 
    biasDerivativeLogistic([1],
                           2,
                           [[1]],
                           [1]),
    g([1],2,[1]) - 1)
test("bias derivative 2", 
biasDerivativeLogistic([1],
                        2,
                        [[1],
                        [2],
                        [3],
                        [4]],
                        [1,0,1,0]),
    ((g([1],2,[1]) - 1) + 
    (g([1],2,[2])) + 
    (g([1],2,[3]) - 1) + 
    (g([1],2,[4])))/4)

function biasDerivativeLogistic (w,b,X,y) {
    /* 
       w is the 1d array of non-bias parameters
       b is the bias parameter
       X is the 2d array of inputs
       y is the 1d array of results
    */

    // find the cost for the bias parameter
    let result = 0;
    for (let i = 0; i < X.length; i++) { // for each observation
        result += g(w,b,X[i]) - y[i]
    }
    return result/X.length
}

test("sigmoid", g([1],0,[1]), 1/(1+Math.exp(-1)))
test("sigmoid", g([-2,5],5,[1,2]), 1/(1+Math.exp(-(-2*1 + 5*2 + 5))))

// sigmoid function
function g(w,b,x) {
    /*
        w is 1d array of non-bias parameters
        b is bias parameter
        x is 1d array: observation
        
        x and w must be same length
    */
   
    // Compute input to sigmoid
   let z = b
   for (let i = 0; i < w.length; i++) {
    z += w[i]*x[i]
   }

   // compute sigmoid
   return 1/(1+Math.exp(-z))
}

function zeroes(arr) { 
    // makes an array of length equal to the given array,
    // but filled with only zeroes (or replace)
    return Array.from("_".repeat(arr.length)).map(() => 0)
}

// calculate the mean of a 2D matrix
// tested
function mean(arr) {
    let result_arr = Array.from(Array.from("".repeat(arr[0].length))).map(() => 0)
    for (let j = 0; j < arr[0].length; j++) { // for each input (column)
        for (let i = 0; i < arr.length; i++){ // for each example (row)
            result_arr[j] = (arr.reduce((sum,nextval) => sum + nextval[j], 0)/arr.length)
            
        }
    }
    return result_arr
}

// normalize a 2D Matrix
// tested
function norm(arr) {
    // store means of each column
    let means = mean(arr)
    // get min and max values
    // by making a list of vectors for each column
    // and calling Math.max/Math.min
    let vectors = zeroes(arr[0])
        .map(i => zeroes(arr))
    for (let j = 0; j < arr[0].length; j++) { // for each column`)
        for (let i = 0; i < arr.length; i++) { // for each row
            vectors[j][i] = arr[i][j]
            
        }
    }
    
    let maxes = vectors.map(lon => Math.max(...lon))
    let mins = vectors.map(lon => Math.min(...lon))
    let resultArrList = zeroes(arr).map(() => zeroes(arr[0]))
    for (let j = 0; j < arr[0].length; j++) { // for each column
        for (let i = 0; i < arr.length; i++) { // for each row
            resultArrList[i][j] = (arr[i][j] - means[j])/(maxes[j] - mins[j])
        }
    }
    return [resultArrList, maxes, mins, means]
}

// normalize new data based on previously-calculated maxes mins and means
function scale(x, maxes, mins, means){
    // x is a 1D vector; a collection of inputs
    // maxes, mins, and means are same length as x; they represent information about the data the model was build with
    let result = zeroes(x)
    for (let i = 0; i < x.length; i++) {
        result[i] = (x[i] - means[i])/(maxes[i] - mins[i])
    }
    return result

}

test("Norm", norm([[1,2],
      [3,4],
      [5,6]])[0],
      [[-0.5,-0.5],[0.0,0.0],[0.5,0.5]])

// debug notes:
// if you assign each element of a list to the same list, mutating any element will mutate them all. Because you're changing the single thing that all of them are pointing to. You'd need to make a copy or repeat the operation to do this cleanly.      

// check that all arguments are equal to each other
function test (message="unnamed", ...a) {
    let checkequal = a[0]
    let result = true
    let errors = []
    for (let i = 0; i < a.length; i++) {
        if (Array.isArray(checkequal)) {
            if (checkArrayEquality(checkequal, a[i])) {

            }
        } else {
            if (checkequal !== a[i]) {
                result = false
                errors = errors.concat(a[i])
            }   
        }
        
    }
    if (result) {
        console.log(`The ${message} test passed!`)
    } else {
        console.log(`The ${message} test failed! The following outputs were not equal to ${checkequal}`, errors);
    }
}

// tested
function checkArrayEquality (a1,a2) {
    if (a1.length !== a2.length) {
        return false
    } else {
        for (let i = 0; i < a1.length; i++) {
            if (a1[i] !== a2[i]) {
                return false
            }
        }
    }
    return true
}

function costLogistic(w,b,X,y) {
    /*
    w is 1D vector of parameters
    b is bias parameter
    X is 2D matrix of inputs
    y is 1D vector of outputs
    */
   let result = 0
   for (let i = 0; i < X.length; i++) {
    const loss = -y[i] * Math.log10(g(w,b,X[i])) + (1 - y[i]) * (Math.log10(1 - g(w,b,X[i])))
    // console.log(`Loss at i = ${i}; w = ${w}; b = ${b}; y[i] = ${y[i]}; X[i] = ${X[i]}: ${loss}. g at this value is ${g(w,b,X[i])}`)
    result += loss
   }
   return result/(X.length)
}

// .load main.ts

// test dataset
// low X values correspond to 0, high X values correspond to 1

let X_test = [
    [1,2],
    [3,4],
    [26,20],
    [1,3],
    [20,21],
    [3,3],
    [3,4]
]
let y_test = [
    0,
    0,
    1,
    0,
    1,
    0,
    0
]
const predictTest = logistic(X_test,y_test,0.01)
console.log("Predictions:")
console.log(predictTest([27,18])) // predicts 1
console.log(predictTest([3,3])) // predicts 0 if correct

console.log("\n\n\n\n\n\n benchmark")

const heartDiseaseDataX = [[4,108],[4,129],[3,187],[2,172],[2,178],[4,160],[4,163],[4,147],[4,155],[4,148],[2,153],[3,142],[2,173],[3,162],[3,174],[2,168],[4,160],[3,139],[2,171],[1,144],[1,162],[2,160],[3,173],[4,132],[3,158],[3,172],[1,114],[4,171],[4,114],[1,151],[4,160],[3,158],[4,161],[3,179],[4,178],[4,120],[4,112],[4,132],[3,137],[4,114],[1,178],[2,162],[3,157],[4,169],[3,165],[3,123],[4,128],[3,157],[3,152],[2,168],[4,140],[4,153],[2,188],[4,144],[4,109],[3,163],[4,158],[3,152],[1,125],[4,142],[3,160],[4,131],[3,170],[4,113],[4,142],[3,155],[3,165],[4,140],[3,147],[3,148],[4,163],[4,99],[4,158],[4,177],[3,151],[4,141],[3,142],[2,180],[4,111],[4,148],[4,143],[3,182],[3,150],[2,172],[3,180],[3,156],[3,115],[4,160],[3,149],[4,151],[4,145],[3,146],[3,175],[3,172],[4,161],[4,142],[4,157],[2,158],[4,186],[4,185],[1,174],[4,159],[3,130],[3,139],[2,156],[4,162],[3,150],[4,140],[4,140],[4,146],[4,144],[1,190],[4,136],[3,97],[2,132],[3,165],[4,182],[4,132],[4,127],[4,150],[4,154],[3,143],[4,111],[1,174],[2,175],[4,133],[4,126],[2,170],[4,163],[3,147],[3,154],[2,202],[4,186],[3,165],[2,161],[4,125],[2,103],[4,130],[3,166],[2,164],[1,159],[2,184],[3,131],[3,154],[3,152],[4,124],[3,179],[2,170],[3,160],[1,178],[4,122],[3,160],[4,145],[4,96],[4,109],[4,173],[4,171],[4,170],[3,151],[2,156],[4,162],[3,158],[4,122],[3,175],[4,168],[3,169],[2,159],[4,156],[2,138],[3,112],[4,111],[4,143],[4,157],[4,132],[4,88],[4,147],[4,105],[3,162],[3,173],[4,166],[4,150],[1,178],[1,145],[4,161],[2,179],[3,194],[2,120],[2,195],[3,146],[3,163],[4,122],[4,143],[4,106],[3,115],[4,125],[1,131],[4,152],[2,162],[1,125],[4,159],[4,154],[3,173],[3,133],[4,161],[4,147],[4,130],[4,126],[2,155],[4,154],[3,170],[1,182],[3,168],[4,165],[4,160],[1,162],[2,172],[4,152],[4,122],[4,182],[3,172],[3,167],[3,179],[4,95],[4,169],[2,192],[4,143],[3,172],[4,108],[4,132],[3,169],[4,117],[3,126],[2,121],[3,163],[4,116],[4,103],[4,144],[2,162],[2,162],[2,153],[2,163],[4,163],[1,145],[3,96],[4,71],[4,156],[4,118],[4,168],[2,140],[4,126],[4,105],[4,105],[3,157],[4,181],[3,173],[4,142],[3,116],[2,143],[2,141],[3,149],[2,152],[1,171],[3,169],[4,125],[4,125],[4,156],[3,134],[4,181],[3,150],[4,138],[4,138],[4,120],[4,125],[1,162],[1,155],[3,152],[3,152],[2,164],[4,131],[4,143],[3,179],[4,130],[2,174],[4,161],[4,140],[4,146],[2,144],[2,163],[2,169],[3,150],[2,166],[4,144],[4,144],[4,136],[2,182],[4,90],[4,123],[1,132],[4,141],[4,115],[2,174],[3,173],]

const heartDiseaseDatay = [1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0,]



const heartPred = logistic(heartDiseaseDataX,heartDiseaseDatay,0.01)

const predictions = heartDiseaseDataX.map(heartPred)

const predictedCorrectly = []
let count = 0

// I know I haven't split the data into training and testing sets. I haven't selected the features with any real 
// purpose either, or done any engineering of them. This is just to prove that the logistic regression I wrote can handle a real situation.

for (let i = 0; i < heartDiseaseDatay.length; i++) {
    if (predictions[i] === heartDiseaseDatay[i]) {
        predictedCorrectly.push(true)
        count++
    } else {
        predictedCorrectly.push(false)
    }
}

console.log(count/heartDiseaseDatay.length)
console.log(predictedCorrectly)