
##############################################################################
class Resolution :
    def __init__(self,width, height):
        self.width = int(width)
        self.height = int( height)
##############################################################################


##############################################################################
def get_image_dimension_from_resolution(resolution):

    if resolution== 'VGA' :
        width=800
        height=480
    if resolution== 'HD' :
        width=720
        height=1080
    return width,height
##############################################################################
