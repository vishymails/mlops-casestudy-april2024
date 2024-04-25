def test_generic() :
    a = 2
    b = 2
    assert a == b


def test_generic1() :
    a = 2
    b = 2
    assert a != b


class NotInRange(Exception) :
    def __init__(self, message="value not in given range- by BVR") :
        self.message = message
        super().__init__(self.message)


def test_generic2() :
    a = 15
    if a not in range(10, 20) :
        raise NotInRange
    

def test_generic3() :
    a = 5
    if a not in range(10, 20) :
        raise NotInRange
    
