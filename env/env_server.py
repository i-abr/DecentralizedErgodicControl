import rospy
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as trans

class Objects(object):

    def __init__(self, p, r, l, w):

        # length, width
        self.l = l
        self.w = w
        self.dim = np.array([l, w])

        trans.rotation_matrix
        self._rot = trans.rotation_matrix(  )

    def is_close(self, x):
        xt = np.abs(np.dot(self._trans, x))
        if (self.dim < xt).any():
            return 1.0
        return 0.

class Map(Visual):

    def __init__(self, num_objs=10):

        self.objects = [
            Object()
        ]

        Visual.__init__(self)


    def check_collision(self, x):
        for obj in self.objects:

def handle_add_two_ints(req):
    print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))
    return AddTwoIntsResponse(req.a + req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()
