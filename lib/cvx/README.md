# CVX

Python wrapper around opencv and numpy for doing image (pre)processing 

## Processing key usage

For example, if your processing key is 'resize_smallest_side-256__central_crop-224-224-3__subtract-123.68,116.78,103.94__transpose-2,0,1', you can create like this:

```
from cvx.img_loader import ImgLoader
from cvx.img_processor import ImgProcessor

img_paths = [...]

img_loader = ImgLoader()
preprocessor = ImgProcessor(
  proc_key = 'resize_smallest_side-256__central_crop-224-224-3__subtract-123.68,116.78,103.94__transpose-2,0,1'
)

data = img_loader.load(img_paths)
prep_data = preprocessor.execute(data)
```
