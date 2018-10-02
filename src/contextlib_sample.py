import contextlib


class MyManager1:
    def __enter__(self):
        print("enter")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")


@contextlib.contextmanager
def my_manager2():
    print("enter")
    yield
    print("exit")


if __name__ == '__main__':
    with MyManager1():
        print("Hello!!")

    print("===================")

    with my_manager2():
        print("Hello!!")