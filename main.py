import random
import numpy as np


p = np.array([0.2,0.4,0.2,0.1,0.1])
matrix = np.array([[3,4,7,0],[5,5,6,1],[8,2,2,6],[3,1,0,7],[1,0,0,9]])

#Bayes criterion

def bayes():
    criterion = np.zeros(np.size(matrix,1))
    sum = 0
      
    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                sum = sum + matrix.transpose()[i][j]*p[j]
            criterion[i] = sum
            sum = 0          
    except Exception as error:
        return f"Error: {error}"

    return criterion
    
def minimumDispersion():
    criterion = np.zeros(np.size(matrix,1))
    sum = 0
    k=0

    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                sum = sum + ((matrix.transpose()[i][j]-bayes()[k])**2)*p[j]
            k=k+1
            
            criterion[i] = sum
            sum = 0          
    except Exception as error:
        return f"Error: {error}"

    return criterion

    
def minimumDespersionModifiedF():
    criterion = np.zeros(np.size(matrix,1))
    sum = 0

    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                sum = sum + ((matrix.transpose()[i][j]-np.average(bayes()))**2)*p[j]
            
            
            criterion[i] = sum
            sum = 0          
    except Exception as error:
        return f"Error: {error}"

    return criterion

def minimumDespersionModifiedS():
    criterion = np.zeros(np.size(matrix,1))
    sum = 0
    

    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                sum = sum + ((matrix.transpose()[i][j]-np.max(bayes()))**2)*p[j]
            
            
            criterion[i] = sum
            sum = 0          
    except Exception as error:
        return f"Error: {error}"

    return criterion


def maximizationProbOfDis():
    criterion = np.zeros(np.size(matrix,1))
    sum = 0
    max_v = np.max(matrix)
    min_v = np.min(matrix)
    rand = 5#random.randint(min_v,max_v)

    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                if matrix[j][i]>=rand:
                    sum = sum + p[j]
            criterion[i] = sum
            sum = 0
               
    except Exception as error:
        return f"Error: {error}"
    
    return criterion

    

def modalProbOfDis():
    max_v_ind = np.argmax(p)

    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                if max_v_ind==j:
                    return np.max(matrix[j])
               
    except Exception as error:
        return f"Error: {error}"
    

def minimumEntropy():
    criterion = np.zeros(np.size(matrix,1))
    prod = np.zeros(np.size(matrix,1))
    prodlog = np.zeros(np.size(matrix,1))
    sum = 0
          
    try:
        
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):
                prod[i] = (p[i]*matrix.transpose()[i][j])/bayes()[i]
                if prod[i]==0:
                    prodlog[i]=0
                else:
                    prodlog[i]=np.log((p[i]*matrix.transpose()[i][j])/bayes()[i])
            criterion[i] = (-1)*(prod[i]+prodlog[i])
            
  
    except Exception as error:
        return f"Error: {error}"

    return criterion

def combinedMethod():

    lambda1 = np.zeros(np.size(matrix,1))
    lambda2 = np.zeros(np.size(matrix,1))
    criterionMin = np.zeros(np.size(matrix,1))
    criterionMax = np.zeros(np.size(matrix,1))
    criterionBetween = np.zeros(np.size(matrix,1))
    sum1 = 0
    sum2 = 0
    
    try:
        for i in range(np.size(matrix,1)):
            for j in range(np.size(matrix,0)):

                sum1 = sum1 + matrix.transpose()[i][j]*p[j]
                sum2 = sum2 + ((matrix.transpose()[i][j])**2)*p[j]             
            lambda1[i] = sum1**2
            lambda2[i] = sum2
            sum1 = sum2 = 0 
        
        l = 0.4#random.randint(0,min(lambda1/lambda2))

        for i in range(np.size(matrix,1)):
            criterionMin[i] = (bayes()[i]**2)*(1-l)-l*minimumDispersion()[i]
        
        l = 0.9#random.randint(max(lambda1/lambda2),2)

        for i in range(np.size(matrix,1)):
            criterionMax[i] = (bayes()[i]**2)*(1-l)-l*minimumDispersion()[i]
        
        l = 0.662#random.randint(min(lambda1/lambda2),max(lambda1/lambda2))

        for i in range(np.size(matrix,1)):
            criterionBetween[i] = (bayes()[i]**2)*(1-l)-l*minimumDispersion()[i]
         
    except Exception as error:
        return f"Error: {error}"

    return f"lambda*: {min(lambda1/lambda2)}\nlambda**: {max(lambda1/lambda2)}\nlambda<lambda*: {max(criterionMin)}\nlambda>lambda*: {max(criterionMax)}\nlambda*<lambda<lambda**: {max(criterionBetween)}"

def main():
    print(bayes())
    print(minimumDispersion())
    print(minimumDespersionModifiedF())
    print(minimumDespersionModifiedS())
    print(maximizationProbOfDis())
    print(modalProbOfDis())
    print(minimumEntropy())
    print(combinedMethod())

if __name__=="__main__":
    main()
