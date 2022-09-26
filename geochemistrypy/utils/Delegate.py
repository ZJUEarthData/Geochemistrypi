# -*- coding: utf-8 -*-

"""
    refer:-----

"""
class Decorator_Base:
    def __init__(self, model):
        self.model = model
        self.model_methods = [f for f in dir(type(self.model)) if not f.startswith('_')]
        self.model_attributes = [a for a in self.model.__dict__.keys()]

    def __getattr__(self, func):
        if func in self.model_methods:
            def method(*args):
                return getattr(self.model, func)(*args)
            return method
        elif func in self.model_attributes:
            return getattr(self.model, func)
        else:
            raise AttributeError

class Decorator:
    def __init__(self, model, callback_methods=[]):
        self.model = model
        self.model_methods = [f for f in dir(type(self.model)) if not f.startswith('_')]
        self.model_attributes = [a for a in self.model.__dict__.keys()]
        self.callback_methods = callback_methods
        self.list_of_callback_methods = []
        self.__divide_callback_functions()

    def __getattr__(self, func):
        if func in self.model_methods and func not in self.list_of_callback_methods:
            def method(*args):
                return getattr(self.model, func)(*args)
            return method
        elif func in self.model_attributes:
            return getattr(self.model, func)
        elif func in self.list_of_callback_methods:
            def method(*args):
                return self.__callback_switchboard(func, *args)
            return method
        else:
            raise AttributeError

    def __divide_callback_functions(self):
        for method in self.callback_methods:
            if 'before' in method.keys(): self.list_of_callback_methods.append(method['before'])
            if 'after' in method.keys(): self.list_of_callback_methods.append(method['after'])

    def __callback_switchboard(self, func, *args):
        for method in self.callback_methods:
            if 'before' in method.keys() and func == method['before']:
                getattr(self, method['do'])()
                getattr(self.model, func)(*args)
            if 'after' in method.keys() and func == method['after']:
                getattr(self.model, func)(*args)
                getattr(self, method['do'])()

class MixinDelegator(object):
    def __getattr__(self, called_method):
        def wrapper(*args, **kwargs):
            delegation_config = getattr(self, 'DELEGATED_METHODS', None)
            if not isinstance(delegation_config, dict):
                raise AttributeError("'%s' object has not defined any delegated methods" % (self.__class__.__name__))
            for delegate_object_str, delegated_methods in delegation_config.items():
                if called_method in delegated_methods:
                    break
                else:
                    raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, called_method))
            delegate_object = getattr(self, delegate_object_str, None)
            return getattr(delegate_object, called_method)(*args, **kwargs)
        return wrapper

class Microwave:
  def __init__(self):
    pass

  def heat_up_food(self):
    print("Food is being microwaved")

class Dishwasher:
  def __init__(self):
    pass

  def wash_dishes(self):
    print("Dishwasher starting")


class Kitchen:
    def __init__(self):
        self.microwave = Microwave()
        self.dishwasher = Dishwasher()
        self.microwave_methods = [f for f in dir(Microwave) if not f.startswith('_')]
        self.dishwasher_methods = [f for f in dir(Dishwasher) if not f.startswith('_')]

    def __getattr__(self, func):
        def method(*args):
            if func in self.microwave_methods:
                return getattr(self.microwave, func)(*args)
            elif func in self.dishwasher_methods:
                return getattr(self.dishwasher, func)(*args)
            else:
                raise AttributeError
        return method

kitchen = Kitchen()
kitchen.heat_up_food()
kitchen.wash_dishes()
