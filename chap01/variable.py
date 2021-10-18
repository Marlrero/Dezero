import numpy as np

# https://www.python.org/dev/peps/pep-0008

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        """
        [변수]

        Args:
            data (np.ndarray): [Numpy 다차원배열]
        """
        self.data = data

if __name__ == '__main__':
    def main():
        data = np.array(1.0)
        x = Variable(data)
        print(x.data)
        
        x.data = np.array(2.0)
        print(x.data)
    
    main()