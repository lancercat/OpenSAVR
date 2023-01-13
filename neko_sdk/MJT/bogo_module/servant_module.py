
class neko_stand_basic:
    def __init__(this,module):
        this.model=module;
    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);
