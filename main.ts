// typescript logistic regression package

// perform logistic regression and return a prediction function
function logistic (X, y, alpha=0.1,iterations=50000,predictionThreshold=0.5) {
    
    /*
        X is a 2d array of inputs
        y is a 1d array of answers
        alpha is the learning rate
        iterations is how many times to run regression
        W is the initial parameters of the model, 1d array
    */

    // normalize each value of X
    let X_norm  = norm(X)


    let w = zeroes(X_norm[0]).map(() => Math.ceil(Math.random()*4))
    let b = Math.ceil(Math.random()*4)

    // begin gradient descent
    for (let n = 0; n < iterations; n++) { // for n iterations
        // make a list of dw_djs for each w
        let dw_dj = costDerivativesLogistic(w,b,X_norm,y)
        let db_dj = biasDerivativeLogistic(w,b,X_norm,y)

        // then update each w with the calculated derivatives
        for (let j = 0; j < w.length; j++) {
            w[j] = w[j] - alpha*dw_dj[j]
        }
        // update bias parameter
        b = b - alpha * db_dj
    }

    // prediction function
    return function (X_new) {
        if (predictionThreshold < g(w,b,X_new)) {
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
    1)

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
        // console.log(`j is ${j}`)
        for (let i = 0; i < arr.length; i++){ // for each example (row)
            // console.log(`i is ${i}`)
            result_arr[j] = (arr.reduce((sum,nextval) => sum + nextval[j], 0)/arr.length)
            // console.log(`result_arr is now ${result_arr}`)
            
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
    // console.log(vectors)
    for (let j = 0; j < arr[0].length; j++) { // for each column
            // console.log(`j is ${j}, so we're accessing the list ${vectors[j]}`)
        for (let i = 0; i < arr.length; i++) { // for each row
            // console.log(`i is ${i}`)
            // console.log(`BEFORE vectors[${j}] is ${vectors[j]}`)
            vectors[j][i] = arr[i][j]
            // console.log(`AFTER vectors[${j}] is ${vectors[j]}`)
            // console.log("\nthis is vectors right now:")
            // console.log(vectors)
            
        }
    }
    
    // console.log(vectors)
    let maxes = vectors.map(lon => Math.max(...lon))
    let mins = vectors.map(lon => Math.min(...lon))
    console.log(maxes)
    console.log(mins)
    // console.log(arr)
    // console.log(zeroes([1,2,3]))
    let resultArrList = zeroes(arr).map(() => zeroes(arr[0]))
    // console.log(resultArrList)
    for (let j = 0; j < arr[0].length; j++) { // for each column
        for (let i = 0; i < arr.length; i++) { // for each row
            resultArrList[i][j] = (arr[i][j] - means[j])/(maxes[j] - mins[j])
        }
    }
    return resultArrList
}




test("Norm", norm([[1,2],
      [3,4],
      [5,6]]),
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

// .load main.ts

// test dataset
// low X values correspond to 0, high X values correspond to 1

let X_test = [
    [1,2],
    [3,4],
    [26,20],
    [1,3],
    [20,21]
]
let y_test = [
    0,
    0,
    1,
    0,
    1
]
const predictTest = logistic(X_test,y_test)
console.log("Predictions:")
console.log(predictTest([27,18]))
console.log(predictTest([0,0]))