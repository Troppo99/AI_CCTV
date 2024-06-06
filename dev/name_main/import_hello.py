from hello import say_hello

print("This is import_hello.py")
say_hello()
print(__name__)

# akan muncul "hello" karena ada fungsi print diluar if pada 'hello.py'