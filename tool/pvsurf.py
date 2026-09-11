from paraview.simple import *
import sys
import os

if len(sys.argv) < 3:
    sys.stderr.write("usage: pvpython pvsurf.py LEVEL[,ZOOM] FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
arg = sys.argv[1].split(",")
level = float(arg[0])
zoom = float(arg[1]) if len(arg) > 1 else 1.15
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
bb = [1e30, -1e30, 1e30, -1e30, 1e30, -1e30]
for path in sys.argv[2:]:
    r = XDMFReader(FileNames=[os.path.splitext(path)[0] + ".body.xdmf2"])
    r.UpdatePipeline()
    b = r.GetDataInformation().GetBounds()
    if b[0] <= b[1]:
        bb = [min(bb[0], b[0]), max(bb[1], b[1]), min(bb[2], b[2]), max(bb[3], b[3]), min(bb[4], b[4]), max(bb[5], b[5])]
    Delete(r)
c = [(bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, (bb[4] + bb[5]) / 2]
L = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
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
    view.GetActiveCamera().Dolly(zoom)
    Render(view)
    SaveScreenshot(base + ".surf.png", view)
    Delete(sd)
    Delete(bd)
    for p in (surfn, smooth, surfs, surf, bodyn, bodys, body):
        Delete(p)
