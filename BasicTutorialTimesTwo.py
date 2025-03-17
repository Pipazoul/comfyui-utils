class TimesTwo:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "input1": ("INT", {}),
            }
        }
        return inputs
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "funcTimesTwo" # <---- look here
    CATEGORY = "pipazoul"

    def funcTimesTwo(self, input1):
        returnval = 0
        returnval = input1 * 2
        return (returnval,)