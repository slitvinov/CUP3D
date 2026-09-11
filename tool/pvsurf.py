from paraview.simple import *
import sys
import os

if len(sys.argv) < 3:
    sys.stderr.write("usage: pvpython pvsurf.py LEVEL FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
level = float(sys.argv[1])
lut = GetColorTransferFunction("u")
lut.ApplyPreset("Cool to Warm", True)
lut.AutomaticRescaleRangeMode = "Never"
m = 0.0
for path in sys.argv[2:]:
    f = "%s.omega%g.xdmf2" % (os.path.splitext(path)[0], level)
    r = XDMFReader(FileNames=[f])
    r.UpdatePipeline()
    a = r.PointData["u"]
    if a is not None:
        lo, hi = a.GetRange()
        m = max(m, abs(lo), abs(hi))
    Delete(r)
lut.RescaleTransferFunction(-m, m)
for path in sys.argv[2:]:
    base = os.path.splitext(path)[0]
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1920, 1080]
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    body = XDMFReader(FileNames=[base + ".body.xdmf2"])
    bodys = ExtractSurface(Input=body)
    bodyn = SurfaceNormals(Input=bodys)
    bodyn.FeatureAngle = 80
    bd = Show(bodyn, view)
    bd.ColorArrayName = ["POINTS", ""]
    bd.DiffuseColor = [0.2, 0.2, 0.2]
    bd.Specular = 0.3
    body.UpdatePipeline()
    if path == sys.argv[2]:
        b = body.GetDataInformation().GetBounds()
        c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        L = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    surf = XDMFReader(FileNames=["%s.omega%g.xdmf2" % (base, level)])
    surf.UpdatePipeline()
    surfs = ExtractSurface(Input=surf)
    smooth = Smooth(Input=surfs)
    smooth.NumberofIterations = 20
    surfn = SurfaceNormals(Input=smooth)
    surfn.FeatureAngle = 80
    sd = Show(surfn, view)
    ColorBy(sd, ("POINTS", "u"))
    sd.LookupTable = lut
    sd.Specular = 0.2
    sd.Opacity = 0.45
    Render(view)
    view.CameraFocalPoint = c
    view.CameraViewUp = [0, 0, 1]
    view.CameraPosition = [c[0] + 0.25 * L, c[1] - 1.0 * L, c[2] + 0.9 * L]
    view.ResetCamera(False)
    view.GetActiveCamera().Dolly(1.6)
    Render(view)
    SaveScreenshot(base + ".surf.png", view)
    Delete(sd)
    Delete(bd)
    for p in (surfn, smooth, surfs, surf, bodyn, bodys, body):
        Delete(p)
