from paraview.simple import *
import sys
import os

if len(sys.argv) < 4:
    sys.stderr.write("usage: pvpython pvsurf.py LEVEL[,ZOOM] PRE FILE.xdmf2 [FILE.xdmf2 ...]\n")
    sys.exit(1)
arg = sys.argv[1].split(",")
tag = "q%g" % float(arg[0][1:]) if arg[0].startswith("q") else "omega%g" % float(arg[0])
zoom = float(arg[1]) if len(arg) > 1 else 1.15
pre = [float(s) for s in sys.argv[2].split(",")]
m = pre[0]
bb = pre[1:7]
ab = pre[7:13]
lut = GetColorTransferFunction("u")
lut.ApplyPreset("Cool to Warm", True)
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(-m, m)
c = [(bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, (bb[4] + bb[5]) / 2]
L = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
for path in sys.argv[3:]:
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
    surf = XDMFReader(FileNames=["%s.%s.xdmf2" % (base, tag)])
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
    view.ResetCamera(ab[0], ab[1], ab[2], ab[3], ab[4], ab[5])
    view.GetActiveCamera().Dolly(zoom)
    Render(view)
    SaveScreenshot(base + ".surf.png", view)
    Delete(sd)
    Delete(bd)
    for p in (surfn, smooth, surfs, surf, bodyn, bodys, body):
        Delete(p)
