A static build links the fast pair list as one relocatable object.
Apple ar was taking the rust archive on its command line as the
archive to write, so the macOS umbrella library was never created.
