import importlib
import sys
import types

import numpy as np


def _install_pyqt_stubs(monkeypatch):
    pyqt5 = types.ModuleType("PyQt5")
    widgets = types.ModuleType("PyQt5.QtWidgets")
    core = types.ModuleType("PyQt5.QtCore")

    class _Signal:
        def connect(self, _callback):
            pass

    class QAction:
        def __init__(self, *_args, **_kwargs):
            self.triggered = _Signal()

    class QWidget:
        pass

    class QDialog:
        Accepted = 1
        Rejected = 0

        def __init__(self, *_args, **_kwargs):
            pass

    class _Widget:
        def __init__(self, *_args, **_kwargs):
            pass

    class QMessageBox:
        Yes = 1
        No = 2
        Cancel = 3

        question_result = Yes

        @staticmethod
        def question(*_args, **_kwargs):
            return QMessageBox.question_result

        @staticmethod
        def warning(*_args, **_kwargs):
            pass

        @staticmethod
        def information(*_args, **_kwargs):
            pass

    class _Qt:
        MoveAction = object()
        UserRole = object()

    widgets.QAction = QAction
    widgets.QCheckBox = _Widget
    widgets.QDialog = QDialog
    widgets.QGridLayout = _Widget
    widgets.QHBoxLayout = _Widget
    widgets.QLabel = _Widget
    widgets.QLineEdit = _Widget
    widgets.QListWidget = _Widget
    widgets.QListWidgetItem = _Widget
    widgets.QMessageBox = QMessageBox
    widgets.QPushButton = _Widget
    widgets.QSpinBox = _Widget
    widgets.QVBoxLayout = _Widget
    widgets.QWidget = QWidget
    core.Qt = _Qt

    monkeypatch.setitem(sys.modules, "PyQt5", pyqt5)
    monkeypatch.setitem(sys.modules, "PyQt5.QtWidgets", widgets)
    monkeypatch.setitem(sys.modules, "PyQt5.QtCore", core)


def _load_transform_module(monkeypatch):
    _install_pyqt_stubs(monkeypatch)
    return importlib.import_module("py4d_browser_plugin.transform.transform")


def _load_datacube_ops_module(monkeypatch):
    _install_pyqt_stubs(monkeypatch)
    return importlib.import_module("py4d_browser_plugin.transform.datacube_ops")


class _Calibration:
    def __init__(self, r_pixel_size=1, q_pixel_size=1, origin=None):
        self.r_pixel_size = r_pixel_size
        self.q_pixel_size = q_pixel_size
        self.origin = origin

    def copy(self):
        return _Calibration(self.r_pixel_size, self.q_pixel_size, self.origin)

    def get_R_pixel_size(self):
        return self.r_pixel_size

    def set_R_pixel_size(self, value):
        self.r_pixel_size = value

    def get_Q_pixel_size(self):
        return self.q_pixel_size

    def set_Q_pixel_size(self, value):
        self.q_pixel_size = value

    def get_origin(self):
        return self.origin

    def set_origin(self, value):
        self.origin = value


class _Datacube:
    def __init__(self, data, calibration=None):
        self.data = data
        self.calibration = calibration or _Calibration()

    def copy(self):
        return _Datacube(self.data.copy(), self.calibration.copy())

    def calibrate(self):
        pass


class _API:
    """Mock of the 1.6 plugin API broker object."""

    def __init__(self, data):
        self.datacube = _Datacube(data)
        self.qt_window = object()
        self.set_datacube_calls = []

    def set_datacube(self, datacube, refresh=True):
        self.set_datacube_calls.append((datacube, refresh))


class _Dialog:
    def __init__(self, values, accepted):
        self._values = values
        self._accepted = accepted

    def exec_(self):
        return self._accepted

    def get_values(self):
        return self._values


class _TransformDialog:
    def __init__(self, operations, accepted):
        self._operations = operations
        self._accepted = accepted

    def exec_(self):
        return self._accepted

    def get_operations(self):
        return self._operations


def _identity_operations():
    return [{"slice": ":", "bin": 1} for _ in range(4)]


def test_axis_permutation_uses_set_datacube(monkeypatch):
    module = _load_transform_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    api = _API(data.copy())
    plugin = object.__new__(module.TransformPlugin)
    plugin.api = api
    plugin.current_permutation = [0, 1, 2, 3]

    new_permutation = [0, 2, 1, 3]
    monkeypatch.setattr(
        module,
        "AxisPermutationDialog",
        lambda *_args: _Dialog(new_permutation, module.QDialog.Accepted),
    )

    plugin.set_axis_permutation()

    np.testing.assert_array_equal(api.datacube.data, np.transpose(data, new_permutation))
    assert plugin.current_permutation == new_permutation
    assert api.set_datacube_calls == [(api.datacube, True)]


def test_diffraction_flips_uses_set_datacube(monkeypatch):
    module = _load_transform_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    api = _API(data.copy())
    plugin = object.__new__(module.TransformPlugin)
    plugin.api = api
    plugin.flip_settings = {"flipud": False, "fliplr": False, "transpose": False}

    new_flips = {"flipud": True, "fliplr": True, "transpose": True}
    monkeypatch.setattr(
        module,
        "DiffractionFlipsDialog",
        lambda *_args: _Dialog(new_flips, module.QDialog.Accepted),
    )

    plugin.set_diffraction_flips()

    expected = np.transpose(np.flip(np.flip(data, axis=2), axis=3), (0, 1, 3, 2))
    np.testing.assert_array_equal(api.datacube.data, expected)
    assert plugin.flip_settings == new_flips
    assert api.set_datacube_calls == [(api.datacube, True)]


def _new_plugin(module, api):
    plugin = object.__new__(module.TransformPlugin)
    plugin.api = api
    plugin.current_permutation = list(module.DEFAULT_PERMUTATION)
    plugin.flip_settings = dict(module.DEFAULT_FLIP_SETTINGS)
    plugin.checkpoints = []
    plugin._checkpoint_ids = iter(range(1, 100))
    return plugin


def test_transform_datacube_cancel_aborts(monkeypatch):
    module = _load_transform_module(monkeypatch)
    api = _API(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
    plugin = _new_plugin(module, api)
    module.QMessageBox.question_result = module.QMessageBox.Cancel

    created = {"called": False}

    def _dialog(*_args):
        created["called"] = True
        return _TransformDialog(_identity_operations(), module.QDialog.Rejected)

    monkeypatch.setattr(module, "TransformDatacubeDialog", _dialog)

    result = plugin.transform_datacube()

    assert result is None
    assert api.set_datacube_calls == []
    assert plugin.checkpoints == []
    assert created["called"] is False


def test_transform_datacube_no_skips_checkpoint_and_transforms(monkeypatch):
    module = _load_transform_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    api = _API(data.copy())
    plugin = _new_plugin(module, api)
    module.QMessageBox.question_result = module.QMessageBox.No

    monkeypatch.setattr(
        module,
        "TransformDatacubeDialog",
        lambda *_args: _TransformDialog(_identity_operations(), module.QDialog.Accepted),
    )

    transformed = plugin.transform_datacube()

    assert transformed is not None
    assert plugin.checkpoints == []
    assert len(api.set_datacube_calls) == 1
    np.testing.assert_array_equal(transformed.data, data)


def test_transform_datacube_yes_creates_checkpoint_then_transforms(monkeypatch):
    module = _load_transform_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    api = _API(data.copy())
    plugin = _new_plugin(module, api)
    module.QMessageBox.question_result = module.QMessageBox.Yes

    operations = [
        {"slice": ":", "bin": 1},
        {"slice": ":", "bin": 1},
        {"slice": ":", "bin": 2},
        {"slice": ":", "bin": 1},
    ]
    monkeypatch.setattr(
        module, "TransformDatacubeDialog", lambda *_args: _TransformDialog(operations, module.QDialog.Accepted)
    )

    transformed = plugin.transform_datacube()

    assert transformed is not None
    assert len(plugin.checkpoints) == 1
    # The checkpoint captured the original (pre-transform) datacube.
    assert plugin.checkpoints[0].shape == data.shape
    np.testing.assert_array_equal(plugin.checkpoints[0].datacube.data, data)
    # The transform was still applied (axis 2 binned by 2).
    assert transformed.data.shape == (2, 3, 2, 5)
    assert len(api.set_datacube_calls) == 1


def test_checkpoint_restore_uses_independent_copies(monkeypatch):
    module = _load_transform_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    api = _API(data.copy())
    api.datacube.calibration.set_R_pixel_size(2)
    plugin = _new_plugin(module, api)
    plugin.current_permutation = [0, 2, 1, 3]
    plugin.flip_settings = {"flipud": True, "fliplr": False, "transpose": False}

    checkpoint = plugin.create_checkpoint("before")
    api.datacube.data[...] = -1
    api.datacube.calibration.set_R_pixel_size(9)

    restored = plugin.restore_checkpoint(checkpoint.checkpoint_id)

    np.testing.assert_array_equal(restored.data, data)
    assert restored is not checkpoint.datacube
    assert restored.calibration is not checkpoint.datacube.calibration
    assert restored.calibration.get_R_pixel_size() == 2
    assert plugin.current_permutation == module.DEFAULT_PERMUTATION
    assert plugin.flip_settings == module.DEFAULT_FLIP_SETTINGS
    assert checkpoint.metadata == {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "estimated_bytes": data.nbytes,
    }
    assert api.set_datacube_calls == [(restored, True)]


def test_checkpoint_delete_rename_replace_and_memory(monkeypatch):
    module = _load_transform_module(monkeypatch)
    api = _API(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
    plugin = _new_plugin(module, api)

    checkpoint = plugin.create_checkpoint("old")
    plugin.rename_checkpoint(checkpoint.checkpoint_id, "renamed")
    assert plugin.checkpoints[0].name == "renamed"
    assert plugin.total_checkpoint_bytes() == api.datacube.data.nbytes

    api.datacube.data = np.zeros((1, 1, 2, 2), dtype=np.float32)
    replacement = plugin.replace_checkpoint(checkpoint.checkpoint_id)
    assert replacement.name == "renamed"
    assert replacement.checkpoint_id == checkpoint.checkpoint_id
    assert replacement.shape == (1, 1, 2, 2)
    assert len(plugin.checkpoints) == 1

    plugin.delete_checkpoint(checkpoint.checkpoint_id)
    assert plugin.checkpoints == []


def test_apply_datacube_operations_slices_and_subsamples_all_axes(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(4 * 5 * 6 * 7).reshape(4, 5, 6, 7)
    datacube = _Datacube(data.copy())

    operations = [
        {"slice": "1:4:2", "bin": 1},
        {"slice": "::2", "bin": 1},
        {"slice": "2:6:2", "bin": 1},
        {"slice": "1:7:3", "bin": 1},
    ]
    transformed = module.apply_datacube_operations(datacube, operations)

    expected = data[1:4:2, ::2, 2:6:2, 1:7:3]
    np.testing.assert_array_equal(transformed.data, expected)
    np.testing.assert_array_equal(datacube.data, data)


def test_apply_datacube_operations_bins_axes_and_updates_pixel_size(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4)
    datacube = _Datacube(data.copy(), _Calibration(r_pixel_size=2, q_pixel_size=3))

    operations = [
        {"slice": ":", "bin": 2},
        {"slice": ":", "bin": 2},
        {"slice": ":", "bin": 2},
        {"slice": ":", "bin": 2},
    ]
    transformed = module.apply_datacube_operations(datacube, operations)

    expected = data
    for axis in range(4):
        shape = list(expected.shape)
        shape[axis : axis + 1] = [shape[axis] // 2, 2]
        expected = expected.reshape(shape).sum(axis=axis + 1)
    np.testing.assert_array_equal(transformed.data, expected)
    assert transformed.calibration.get_R_pixel_size() == 4
    assert transformed.calibration.get_Q_pixel_size() == 6
    assert datacube.calibration.get_R_pixel_size() == 2
    assert datacube.calibration.get_Q_pixel_size() == 3


def test_apply_datacube_operations_adjusts_q_origin_for_slices(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(3 * 3 * 6 * 6).reshape(3, 3, 6, 6)
    datacube = _Datacube(data.copy(), _Calibration(q_pixel_size=0.5, origin=(10.0, 20.0)))

    transformed = module.apply_datacube_operations(
        datacube,
        [
            {"slice": ":", "bin": 1},
            {"slice": ":", "bin": 1},
            {"slice": "2:", "bin": 1},
            {"slice": "4:", "bin": 1},
        ],
    )

    assert transformed.calibration.get_origin() == (8.0, 16.0)
    assert datacube.calibration.get_origin() == (10.0, 20.0)


def test_apply_datacube_operations_adjusts_q_origin_for_binning(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(2 * 2 * 8 * 8).reshape(2, 2, 8, 8)
    datacube = _Datacube(data.copy(), _Calibration(q_pixel_size=0.5, origin=(6.0, 8.0)))

    transformed = module.apply_datacube_operations(
        datacube,
        [
            {"slice": ":", "bin": 1},
            {"slice": ":", "bin": 1},
            {"slice": "2:", "bin": 2},
            {"slice": "0:", "bin": 2},
        ],
    )

    # New origin = (old_origin - crop_start) / (step * bin), in the
    # transformed array's own pixel grid.
    assert transformed.calibration.get_origin() == (2.0, 4.0)
    assert datacube.calibration.get_origin() == (6.0, 8.0)


def test_apply_datacube_operations_preserves_pixel_size_for_asymmetric_spacing(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(4 * 4 * 6 * 6).reshape(4, 4, 6, 6)
    datacube = _Datacube(data.copy(), _Calibration(r_pixel_size=2, q_pixel_size=3))

    transformed = module.apply_datacube_operations(
        datacube,
        [
            {"slice": "::2", "bin": 1},
            {"slice": ":", "bin": 1},
            {"slice": ":", "bin": 2},
            {"slice": ":", "bin": 1},
        ],
    )

    assert transformed.calibration.get_R_pixel_size() == 2
    assert transformed.calibration.get_Q_pixel_size() == 3


class _Root:
    def __init__(self, name):
        self.name = name


class _DatacubeWithRoot(_Datacube):
    def __init__(self, data, calibration=None, root_name="datacube_root"):
        super().__init__(data, calibration)
        self.root = _Root(root_name)

    def copy(self):
        # Mimic py4DSTEM's DataCube.copy(): rebuilding via the constructor
        # gives the new object a differently-named root.
        copied = _DatacubeWithRoot(
            self.data.copy(), self.calibration.copy(), root_name="py4DSTEM_root"
        )
        return copied


def test_copy_datacube_preserves_original_root_name(monkeypatch):
    _install_pyqt_stubs(monkeypatch)
    module = importlib.import_module("py4d_browser_plugin.transform.checkpoints")

    datacube = _DatacubeWithRoot(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2))
    assert datacube.root.name == "datacube_root"

    copied = module.copy_datacube(datacube)

    assert copied.root.name == "datacube_root"
    assert datacube.root.name == "datacube_root"


def test_apply_datacube_operations_invalid_input_does_not_mutate(monkeypatch):
    module = _load_datacube_ops_module(monkeypatch)
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    datacube = _Datacube(data.copy())

    try:
        module.apply_datacube_operations(
            datacube,
            [
                {"slice": ":", "bin": 1},
                {"slice": ":", "bin": 0},
                {"slice": ":", "bin": 1},
                {"slice": ":", "bin": 1},
            ],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid bin factor to raise ValueError")

    np.testing.assert_array_equal(datacube.data, data)
