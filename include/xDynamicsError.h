#ifndef XDYNAMICSERROR_H
#define XDYNAMICSERROR_H

//#include "xdynamics_manager/xdynamics_manager_decl.h"
#include "xdynamics_decl.h"
#include "xdynamics_global.h"
#include <cstring>

#define ERROR_DETECTED -99

class XDYNAMICS_API xDynamicsError
{
public:
	enum
	{
		xdynamicsSuccess = 0,
		xdynamicsErrorMultiBodySimulationHHTIterationOver = 1,
		xdynamicsErrorLinearEquationCalculation = 2,
		xdynamicsErrorMultiBodyModelInitialization = 3,
		xdynamicsErrorDiscreteElementMethodModelInitialization = 4,
		xdynamicsErrorMultiBodyModelRedundantCondition = 5,
		xdynamicsErrorIncompressibleSPHInitialization = 6,
		xdynamicsErrorExcelModelingData = 7,

		/**
	 * The API call returned with no errors. In the case of query calls, this
	 * also means that the operation being queried is complete (see
	 * ::xCudaEventQuery() and ::xCudaStreamQuery()).
	 */
		xCudaSuccess = 90,

		/**
		 * This indicates that one or more of the parameters passed to the API call
		 * is not within an acceptable range of values.
		 */
		xCudaErrorInvalidValue = 91,

		/**
		 * The API call failed because it was unable to allocate enough memory to
		 * perform the requested operation.
		 */
		xCudaErrorMemoryAllocation = 92,

		/**
		 * The API call failed because the xCuda driver and runtime could not be
		 * initialized.
		 */
		xCudaErrorInitializationError = 93,

		/**
		 * This indicates that a xCuda Runtime API call cannot be executed because
		 * it is being called during process shut down, at a point in time after
		 * xCuda driver has been unloaded.
		 */
		xCudaErrorxCudartUnloading = 94,

		/**
		 * This indicates profiler is not initialized for this run. This can
		 * happen when the application is running with external profiling tools
		 * like visual profiler.
		 */
		xCudaErrorProfilerDisabled = 95,

		/**
		 * \deprecated
		 * This error return is deprecated as of xCuda 5.0. It is no longer an error
		 * to attempt to enable/disable the profiling via ::xCudaProfilerStart or
		 * ::xCudaProfilerStop without initialization.
		 */
		xCudaErrorProfilerNotInitialized = 96,

		/**
		 * \deprecated
		 * This error return is deprecated as of xCuda 5.0. It is no longer an error
		 * to call xCudaProfilerStart() when profiling is already enabled.
		 */
		xCudaErrorProfilerAlreadyStarted = 97,

		/**
		 * \deprecated
		 * This error return is deprecated as of xCuda 5.0. It is no longer an error
		 * to call xCudaProfilerStop() when profiling is already disabled.
		 */
		xCudaErrorProfilerAlreadyStopped = 98,

		/**
		 * This indicates that a kernel launch is requesting resources that can
		 * never be satisfied by the current device. Requesting more shared memory
		 * per block than the device supports will trigger this error, as will
		 * requesting too many threads or blocks. See ::xCudaDeviceProp for more
		 * device limitations.
		 */
		xCudaErrorInvalidConfiguration = 99,

		/**
		 * This indicates that one or more of the pitch-related parameters passed
		 * to the API call is not within the acceptable range for pitch.
		 */
		xCudaErrorInvalidPitchValue = 912,

		/**
		 * This indicates that the symbol name/identifier passed to the API call
		 * is not a valid name or identifier.
		 */
		xCudaErrorInvalidSymbol = 913,

		/**
		 * This indicates that at least one host pointer passed to the API call is
		 * not a valid host pointer.
		 * \deprecated
		 * This error return is deprecated as of xCuda 10.1.
		 */
		xCudaErrorInvalidHostPointer = 916,

		/**
		 * This indicates that at least one device pointer passed to the API call is
		 * not a valid device pointer.
		 * \deprecated
		 * This error return is deprecated as of xCuda 10.1.
		 */
		xCudaErrorInvalidDevicePointer = 917,

		/**
		 * This indicates that the texture passed to the API call is not a valid
		 * texture.
		 */
		xCudaErrorInvalidTexture = 918,

		/**
		 * This indicates that the texture binding is not valid. This occurs if you
		 * call ::xCudaGetTextureAlignmentOffset() with an unbound texture.
		 */
		xCudaErrorInvalidTextureBinding = 919,

		/**
		 * This indicates that the channel descriptor passed to the API call is not
		 * valid. This occurs if the format is not one of the formats specified by
		 * ::xCudaChannelFormatKind, or if one of the dimensions is invalid.
		 */
		xCudaErrorInvalidChannelDescriptor = 920,

		/**
		 * This indicates that the direction of the memcpy passed to the API call is
		 * not one of the types specified by ::xCudaMemcpyKind.
		 */
		xCudaErrorInvalidMemcpyDirection = 921,

		/**
		 * This indicated that the user has taken the address of a constant variable,
		 * which was forbidden up until the xCuda 3.1 release.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Variables in constant
		 * memory may now have their address taken by the runtime via
		 * ::xCudaGetSymbolAddress().
		 */
		xCudaErrorAddressOfConstant = 922,

		/**
		 * This indicated that a texture fetch was not able to be performed.
		 * This was previously used for device emulation of texture operations.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorTextureFetchFailed = 923,

		/**
		 * This indicated that a texture was not bound for access.
		 * This was previously used for device emulation of texture operations.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorTextureNotBound = 924,

		/**
		 * This indicated that a synchronization operation had failed.
		 * This was previously used for some device emulation functions.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorSynchronizationError = 925,

		/**
		 * This indicates that a non-float texture was being accessed with linear
		 * filtering. This is not supported by xCuda.
		 */
		xCudaErrorInvalidFilterSetting = 926,

		/**
		 * This indicates that an attempt was made to read a non-float texture as a
		 * normalized float. This is not supported by xCuda.
		 */
		xCudaErrorInvalidNormSetting = 927,

		/**
		 * Mixing of device and device emulation code was not allowed.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorMixedDeviceExecution = 928,

		/**
		 * This indicates that the API call is not yet implemented. Production
		 * releases of xCuda will never return this error.
		 * \deprecated
		 * This error return is deprecated as of xCuda 4.1.
		 */
		xCudaErrorNotYetImplemented = 931,

		/**
		 * This indicated that an emulated device pointer exceeded the 32-bit address
		 * range.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorMemoryValueTooLarge = 932,

		/**
		 * This indicates that the installed NVIDIA xCuda driver is older than the
		 * xCuda runtime library. This is not a supported configuration. Users should
		 * install an updated NVIDIA display driver to allow the application to run.
		 */
		xCudaErrorInsufficientDriver = 935,

		/**
		 * This indicates that the surface passed to the API call is not a valid
		 * surface.
		 */
		xCudaErrorInvalidSurface = 937,

		/**
		 * This indicates that multiple global or constant variables (across separate
		 * xCuda source files in the application) share the same string name.
		 */
		xCudaErrorDuplicateVariableName = 943,

		/**
		 * This indicates that multiple textures (across separate xCuda source
		 * files in the application) share the same string name.
		 */
		xCudaErrorDuplicateTextureName = 944,

		/**
		 * This indicates that multiple surfaces (across separate xCuda source
		 * files in the application) share the same string name.
		 */
		xCudaErrorDuplicateSurfaceName = 945,

		/**
		 * This indicates that all xCuda devices are busy or unavailable at the current
		 * time. Devices are often busy/unavailable due to use of
		 * ::xCudaComputeModeExclusive, ::xCudaComputeModeProhibited or when long
		 * running xCuda kernels have filled up the GPU and are blocking new work
		 * from starting. They can also be unavailable due to memory constraints
		 * on a device that already has active xCuda work being performed.
		 */
		xCudaErrorDevicesUnavailable = 946,

		/**
		 * This indicates that the current context is not compatible with this
		 * the xCuda Runtime. This can only occur if you are using xCuda
		 * Runtime/Driver interoperability and have created an existing Driver
		 * context using the driver API. The Driver context may be incompatible
		 * either because the Driver context was created using an older version
		 * of the API, because the Runtime API call expects a primary driver
		 * context and the Driver context is not primary, or because the Driver
		 * context has been destroyed. Please see \ref xCudaRT_DRIVER "Interactions
		 * with the xCuda Driver API" for more information.
		 */
		xCudaErrorIncompatibleDriverContext = 949,

		/**
		 * The device function being invoked (usually via ::xCudaLaunchKernel()) was not
		 * previously configured via the ::xCudaConfigureCall() function.
		 */
		xCudaErrorMissingConfiguration = 952,

		/**
		 * This indicated that a previous kernel launch failed. This was previously
		 * used for device emulation of kernel launches.
		 * \deprecated
		 * This error return is deprecated as of xCuda 3.1. Device emulation mode was
		 * removed with the xCuda 3.1 release.
		 */
		xCudaErrorPriorLaunchFailure = 953,

		/**
		 * This error indicates that a device runtime grid launch did not occur
		 * because the depth of the child grid would exceed the maximum supported
		 * number of nested grid launches.
		 */
		xCudaErrorLaunchMaxDepthExceeded = 965,

		/**
		 * This error indicates that a grid launch did not occur because the kernel
		 * uses file-scoped textures which are unsupported by the device runtime.
		 * Kernels launched via the device runtime only support textures created with
		 * the Texture Object API's.
		 */
		xCudaErrorLaunchFileScopedTex = 966,

		/**
		 * This error indicates that a grid launch did not occur because the kernel
		 * uses file-scoped surfaces which are unsupported by the device runtime.
		 * Kernels launched via the device runtime only support surfaces created with
		 * the Surface Object API's.
		 */
		xCudaErrorLaunchFileScopedSurf = 967,

		/**
		 * This error indicates that a call to ::xCudaDeviceSynchronize made from
		 * the device runtime failed because the call was made at grid depth greater
		 * than than either the default (2 levels of grids) or user specified device
		 * limit ::xCudaLimitDevRuntimeSyncDepth. To be able to synchronize on
		 * launched grids at a greater depth successfully, the maximum nested
		 * depth at which ::xCudaDeviceSynchronize will be called must be specified
		 * with the ::xCudaLimitDevRuntimeSyncDepth limit to the ::xCudaDeviceSetLimit
		 * api before the host-side launch of a kernel using the device runtime.
		 * Keep in mind that additional levels of sync depth require the runtime
		 * to reserve large amounts of device memory that cannot be used for
		 * user allocations.
		 */
		xCudaErrorSyncDepthExceeded = 968,

		/**
		 * This error indicates that a device runtime grid launch failed because
		 * the launch would exceed the limit ::xCudaLimitDevRuntimePendingLaunchCount.
		 * For this launch to proceed successfully, ::xCudaDeviceSetLimit must be
		 * called to set the ::xCudaLimitDevRuntimePendingLaunchCount to be higher
		 * than the upper bound of outstanding launches that can be issued to the
		 * device runtime. Keep in mind that raising the limit of pending device
		 * runtime launches will require the runtime to reserve device memory that
		 * cannot be used for user allocations.
		 */
		xCudaErrorLaunchPendingCountExceeded = 969,

		/**
		 * The requested device function does not exist or is not compiled for the
		 * proper device architecture.
		 */
		xCudaErrorInvalidDeviceFunction = 998,

		/**
		 * This indicates that no xCuda-capable devices were detected by the installed
		 * xCuda driver.
		 */
		xCudaErrorNoDevice = 9100,

		/**
		 * This indicates that the device ordinal supplied by the user does not
		 * correspond to a valid xCuda device.
		 */
		xCudaErrorInvalidDevice = 9101,

		/**
		 * This indicates an internal startup failure in the xCuda runtime.
		 */
		xCudaErrorStartupFailure = 9127,

		/**
		 * This indicates that the device kernel image is invalid.
		 */
		xCudaErrorInvalidKernelImage = 9200,

		/**
		 * This most frequently indicates that there is no context bound to the
		 * current thread. This can also be returned if the context passed to an
		 * API call is not a valid handle (such as a context that has had
		 * ::cuCtxDestroy() invoked on it). This can also be returned if a user
		 * mixes different API versions (i.e. 3010 context with 3020 API calls).
		 * See ::cuCtxGetApiVersion() for more details.
		 */
		xCudaErrorDeviceUninitilialized = 9201,

		/**
		 * This indicates that the buffer object could not be mapped.
		 */
		xCudaErrorMapBufferObjectFailed = 9205,

		/**
		 * This indicates that the buffer object could not be unmapped.
		 */
		xCudaErrorUnmapBufferObjectFailed = 9206,

		/**
		 * This indicates that the specified array is currently mapped and thus
		 * cannot be destroyed.
		 */
		xCudaErrorArrayIsMapped = 9207,

		/**
		 * This indicates that the resource is already mapped.
		 */
		xCudaErrorAlreadyMapped = 9208,

		/**
		 * This indicates that there is no kernel image available that is suitable
		 * for the device. This can occur when a user specifies code generation
		 * options for a particular xCuda source file that do not include the
		 * corresponding device configuration.
		 */
		xCudaErrorNoKernelImageForDevice = 9209,

		/**
		 * This indicates that a resource has already been acquired.
		 */
		xCudaErrorAlreadyAcquired = 9210,

		/**
		 * This indicates that a resource is not mapped.
		 */
		xCudaErrorNotMapped = 9211,

		/**
		 * This indicates that a mapped resource is not available for access as an
		 * array.
		 */
		xCudaErrorNotMappedAsArray = 9212,

		/**
		 * This indicates that a mapped resource is not available for access as a
		 * pointer.
		 */
		xCudaErrorNotMappedAsPointer = 9213,

		/**
		 * This indicates that an uncorrectable ECC error was detected during
		 * execution.
		 */
		xCudaErrorECCUncorrectable = 9214,

		/**
		 * This indicates that the ::xCudaLimit passed to the API call is not
		 * supported by the active device.
		 */
		xCudaErrorUnsupportedLimit = 9215,

		/**
		 * This indicates that a call tried to access an exclusive-thread device that
		 * is already in use by a different thread.
		 */
		xCudaErrorDeviceAlreadyInUse = 9216,

		/**
		 * This error indicates that P2P access is not supported across the given
		 * devices.
		 */
		xCudaErrorPeerAccessUnsupported = 9217,

		/**
		 * A PTX compilation failed. The runtime may fall back to compiling PTX if
		 * an application does not contain a suitable binary for the current device.
		 */
		xCudaErrorInvalidPtx = 9218,

		/**
		 * This indicates an error with the OpenGL or DirectX context.
		 */
		xCudaErrorInvalidGraphicsContext = 9219,

		/**
		 * This indicates that an uncorrectable NVLink error was detected during the
		 * execution.
		 */
		xCudaErrorNvlinkUncorrectable = 9220,

		/**
		 * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
		 * library is used for PTX compilation. The runtime may fall back to compiling PTX
		 * if an application does not contain a suitable binary for the current device.
		 */
		xCudaErrorJitCompilerNotFound = 9221,

		/**
		 * This indicates that the device kernel source is invalid.
		 */
		xCudaErrorInvalidSource = 9300,

		/**
		 * This indicates that the file specified was not found.
		 */
		xCudaErrorFileNotFound = 9301,

		/**
		 * This indicates that a link to a shared object failed to resolve.
		 */
		xCudaErrorSharedObjectSymbolNotFound = 9302,

		/**
		 * This indicates that initialization of a shared object failed.
		 */
		xCudaErrorSharedObjectInitFailed = 9303,

		/**
		 * This error indicates that an OS call failed.
		 */
		xCudaErrorOperatingSystem = 9304,

		/**
		 * This indicates that a resource handle passed to the API call was not
		 * valid. Resource handles are opaque types like ::xCudaStream_t and
		 * ::xCudaEvent_t.
		 */
		xCudaErrorInvalidResourceHandle = 9400,

		/**
		 * This indicates that a resource required by the API call is not in a
		 * valid state to perform the requested operation.
		 */
		xCudaErrorIllegalState = 9401,

		/**
		 * This indicates that a named symbol was not found. Examples of symbols
		 * are global/constant variable names, texture names, and surface names.
		 */
		xCudaErrorSymbolNotFound = 9500,

		/**
		 * This indicates that asynchronous operations issued previously have not
		 * completed yet. This result is not actually an error, but must be indicated
		 * differently than ::xCudaSuccess (which indicates completion). Calls that
		 * may return this value include ::xCudaEventQuery() and ::xCudaStreamQuery().
		 */
		xCudaErrorNotReady = 9600,

		/**
		 * The device encountered a load or store instruction on an invalid memory address.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorIllegalAddress = 9700,

		/**
		 * This indicates that a launch did not occur because it did not have
		 * appropriate resources. Although this error is similar to
		 * ::xCudaErrorInvalidConfiguration, this error usually indicates that the
		 * user has attempted to pass too many arguments to the device kernel, or the
		 * kernel launch specifies too many threads for the kernel's register count.
		 */
		xCudaErrorLaunchOutOfResources = 9701,

		/**
		 * This indicates that the device kernel took too long to execute. This can
		 * only occur if timeouts are enabled - see the device property
		 * \ref ::xCudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
		 * for more information.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorLaunchTimeout = 9702,

		/**
		 * This error indicates a kernel launch that uses an incompatible texturing
		 * mode.
		 */
		xCudaErrorLaunchIncompatibleTexturing = 9703,

		/**
		 * This error indicates that a call to ::xCudaDeviceEnablePeerAccess() is
		 * trying to re-enable peer addressing on from a context which has already
		 * had peer addressing enabled.
		 */
		xCudaErrorPeerAccessAlreadyEnabled = 9704,

		/**
		 * This error indicates that ::xCudaDeviceDisablePeerAccess() is trying to
		 * disable peer addressing which has not been enabled yet via
		 * ::xCudaDeviceEnablePeerAccess().
		 */
		xCudaErrorPeerAccessNotEnabled = 9705,

		/**
		 * This indicates that the user has called ::xCudaSetValidDevices(),
		 * ::xCudaSetDeviceFlags(), ::xCudaD3D9SetDirect3DDevice(),
		 * ::xCudaD3D10SetDirect3DDevice, ::xCudaD3D11SetDirect3DDevice(), or
		 * ::xCudaVDPAUSetVDPAUDevice() after initializing the xCuda runtime by
		 * calling non-device management operations (allocating memory and
		 * launching kernels are examples of non-device management operations).
		 * This error can also be returned if using runtime/driver
		 * interoperability and there is an existing ::CUcontext active on the
		 * host thread.
		 */
		xCudaErrorSetOnActiveProcess = 9708,

		/**
		 * This error indicates that the context current to the calling thread
		 * has been destroyed using ::cuCtxDestroy, or is a primary context which
		 * has not yet been initialized.
		 */
		xCudaErrorContextIsDestroyed = 9709,

		/**
		 * An assert triggered in device code during kernel execution. The device
		 * cannot be used again. All existing allocations are invalid. To continue
		 * using xCuda, the process must be terminated and relaunched.
		 */
		xCudaErrorAssert = 9710,

		/**
		 * This error indicates that the hardware resources required to enable
		 * peer access have been exhausted for one or more of the devices
		 * passed to ::xCudaEnablePeerAccess().
		 */
		xCudaErrorTooManyPeers = 9711,

		/**
		 * This error indicates that the memory range passed to ::xCudaHostRegister()
		 * has already been registered.
		 */
		xCudaErrorHostMemoryAlreadyRegistered = 9712,

		/**
		 * This error indicates that the pointer passed to ::xCudaHostUnregister()
		 * does not correspond to any currently registered memory region.
		 */
		xCudaErrorHostMemoryNotRegistered = 9713,

		/**
		 * Device encountered an error in the call stack during kernel execution,
		 * possibly due to stack corruption or exceeding the stack size limit.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorHardwareStackError = 9714,

		/**
		 * The device encountered an illegal instruction during kernel execution
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorIllegalInstruction = 9715,

		/**
		 * The device encountered a load or store instruction
		 * on a memory address which is not aligned.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorMisalignedAddress = 9716,

		/**
		 * While executing a kernel, the device encountered an instruction
		 * which can only operate on memory locations in certain address spaces
		 * (global, shared, or local), but was supplied a memory address not
		 * belonging to an allowed address space.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorInvalidAddressSpace = 9717,

		/**
		 * The device encountered an invalid program counter.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorInvalidPc = 9718,

		/**
		 * An exception occurred on the device while executing a kernel. Common
		 * causes include dereferencing an invalid device pointer and accessing
		 * out of bounds shared memory. Less common cases can be system specific - more
		 * information about these cases can be found in the system specific user guide.
		 * This leaves the process in an inconsistent state and any further xCuda work
		 * will return the same error. To continue using xCuda, the process must be terminated
		 * and relaunched.
		 */
		xCudaErrorLaunchFailure = 9719,

		/**
		 * This error indicates that the number of blocks launched per grid for a kernel that was
		 * launched via either ::xCudaLaunchCooperativeKernel or ::xCudaLaunchCooperativeKernelMultiDevice
		 * exceeds the maximum number of blocks as allowed by ::xCudaOccupancyMaxActiveBlocksPerMultiprocessor
		 * or ::xCudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
		 * as specified by the device attribute ::xCudaDevAttrMultiProcessorCount.
		 */
		xCudaErrorCooperativeLaunchTooLarge = 9720,

		/**
		 * This error indicates the attempted operation is not permitted.
		 */
		xCudaErrorNotPermitted = 9800,

		/**
		 * This error indicates the attempted operation is not supported
		 * on the current system or device.
		 */
		xCudaErrorNotSupported = 9801,

		/**
		 * This error indicates that the system is not yet ready to start any xCuda
		 * work.  To continue using xCuda, verify the system configuration is in a
		 * valid state and all required driver daemons are actively running.
		 * More information about this error can be found in the system specific
		 * user guide.
		 */
		xCudaErrorSystemNotReady = 9802,

		/**
		 * This error indicates that there is a mismatch between the versions of
		 * the display driver and the xCuda driver. Refer to the compatibility documentation
		 * for supported versions.
		 */
		xCudaErrorSystemDriverMismatch = 9803,

		/**
		 * This error indicates that the system was upgraded to run with forward compatibility
		 * but the visible hardware detected by xCuda does not support this configuration.
		 * Refer to the compatibility documentation for the supported hardware matrix or ensure
		 * that only supported hardware is visible during initialization via the xCuda_VISIBLE_DEVICES
		 * environment variable.
		 */
		xCudaErrorCompatNotSupportedOnDevice = 9804,

		/**
		 * The operation is not permitted when the stream is capturing.
		 */
		xCudaErrorStreamCaptureUnsupported = 9900,

		/**
		 * The current capture sequence on the stream has been invalidated due to
		 * a previous error.
		 */
		xCudaErrorStreamCaptureInvalidated = 9901,

		/**
		 * The operation would have resulted in a merge of two independent capture
		 * sequences.
		 */
		xCudaErrorStreamCaptureMerge = 9902,

		/**
		 * The capture was not initiated in this stream.
		 */
		xCudaErrorStreamCaptureUnmatched = 9903,

		/**
		 * The capture sequence contains a fork that was not joined to the primary
		 * stream.
		 */
		xCudaErrorStreamCaptureUnjoined = 9904,

		/**
		 * A dependency would have been created which crosses the capture sequence
		 * boundary. Only implicit in-stream ordering dependencies are allowed to
		 * cross the boundary.
		 */
		xCudaErrorStreamCaptureIsolation = 9905,

		/**
		 * The operation would have resulted in a disallowed implicit dependency on
		 * a current capture sequence from xCudaStreamLegacy.
		 */
		xCudaErrorStreamCaptureImplicit = 9906,

		/**
		 * The operation is not permitted on an event which was last recorded in a
		 * capturing stream.
		 */
		xCudaErrorCapturedEvent = 9907,

		/**
		 * A stream capture sequence not initiated with the ::xCudaStreamCaptureModeRelaxed
		 * argument to ::xCudaStreamBeginCapture was passed to ::xCudaStreamEndCapture in a
		 * different thread.
		 */
		xCudaErrorStreamCaptureWrongThread = 9908,

		/**
		 * This indicates that an unknown internal error has occurred.
		 */
		xCudaErrorUnknown = 9999,

		/**
		 * Any unhandled xCuda driver error is added to this value and returned via
		 * the runtime. Production releases of xCuda should not return such errors.
		 * \deprecated
		 * This error return is deprecated as of xCuda 4.1.
		 */
		xCudaErrorApiFailureBase = 910000
	};
	xDynamicsError();
	~xDynamicsError();
	static char* getErrorString();
	//static void checkXerror(int val);
	static bool _check(int result, char const *const func, const char* const file, int const line);
private:
	static char *_xdynamicsGetErrorEnum(int error);
	static char err[255];
};

#define checkXerror(val) xDynamicsError::_check ( (val), #val, __FILE__, __LINE__ )

#endif