

def create_new_field_name(field_name, prefix="", suffix=""):
    """
    Creates a new field name in the format that is used across different steps.
    """
    fn = []
    if prefix:
        fn.append(prefix)
    fn.append(field_name)
    if suffix:
        fn.append(suffix)
    return "__".join(fn)
